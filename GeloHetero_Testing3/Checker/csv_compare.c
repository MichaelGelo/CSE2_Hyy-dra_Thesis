/**
 * @file csv_compare.c
 * @brief Correctness Verification Tool for Damerau-Levenshtein Implementations
 *
 * Compares three implementation results against the C CPU reference:
 *   (1)  C  vs  Homo FPGA
 *   (2)  C  vs  Homo GPU
 *   (3)  C  vs  Hetero (GPU + FPGA)
 *
 * Rows are matched by key: (Query File, Reference File).
 * Index lists are compared as sets (order-independent).
 * Lengths (Query Length, Reference Length) are intentionally skipped.
 *
 * Build:
 *   gcc -o csv_compare csv_compare.c -std=c99 -Wall -Wextra && ./csv_compare
 * Run:
 *   ./csv_compare
 */

#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * FILE PATH CONFIGURATION
 * Edit these paths before running.
 * ============================================================================ */
#define CSV_C      "/home/dlsu-cse/Downloads/Testing2026/Testing/finalizing/Cresults.csv"
#define CSV_FPGA   "/home/dlsu-cse/Downloads/Testing2026/Testing/finalizing/FPGAresults.csv"
#define CSV_GPU    "/home/dlsu-cse/Downloads/Testing2026/Testing/finalizing/HomoGPUresults.csv"
#define CSV_HETERO "/home/dlsu-cse/Downloads/Testing2026/Testing/finalizing/Heteroresults.csv"

/* ============================================================================
 * BUFFER SIZES
 * LINE_BUF  : maximum length of one CSV row (handles very large index lists)
 * FIELD_BUF : maximum length of one parsed field
 * ============================================================================ */
#define LINE_BUF  (1 << 20)   /* 1 MB  */
#define FIELD_BUF (1 << 19)   /* 512 KB */

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

typedef struct {
    char *query_file;       /* col 0 – key part 1                        */
    char *ref_file;         /* col 2 – key part 2                        */
    int   num_hits;         /* col 4                                     */
    char *hit_indexes;      /* col 5 – comma-separated list or "N/A"     */
    char *lowest_score;     /* col 6 – integer string or "N/A"           */
    char *lowest_indexes;   /* col 7 – comma-separated list or "N/A"     */
    int   last_score;       /* col 8                                     */
} CsvRow;

typedef struct {
    CsvRow *rows;
    int     count;
    int     capacity;
    char    label[128];     /* first line of the CSV, e.g. "C Run"       */
} CsvFile;

typedef struct {
    int missing;            /* row present in C but absent in other file */
    int num_hits;
    int hit_indexes;
    int lowest_score;
    int lowest_indexes;
    int last_score;
} FieldMismatch;

typedef struct {
    CsvRow       *crow;     /* pointer into C file                       */
    CsvRow       *orow;     /* pointer into other file (NULL if missing) */
    FieldMismatch fm;
} MismatchEntry;

/* ============================================================================
 * CSV PARSING
 * ============================================================================ */

/**
 * @brief Parse one CSV field, handling RFC-4180 quoting.
 * @param p       Pointer to start of field in the line buffer.
 * @param out     Output buffer for the unquoted field content.
 * @param out_sz  Size of output buffer.
 * @return Pointer to the start of the next field (after the separator comma).
 */
static const char *parse_field(const char *p, char *out, int out_sz)
{
    int i = 0;
    if (*p == '"') {
        p++;    /* skip opening quote */
        while (*p && *p != '"') {
            if (i < out_sz - 1) out[i++] = *p;
            p++;
        }
        if (*p == '"') p++;   /* skip closing quote */
        if (*p == ',') p++;   /* skip separator     */
    } else {
        while (*p && *p != ',' && *p != '\n' && *p != '\r') {
            if (i < out_sz - 1) out[i++] = *p;
            p++;
        }
        if (*p == ',') p++;   /* skip separator */
    }
    out[i] = '\0';
    return p;
}

/**
 * @brief Load and parse a CSV results file.
 *        Skips the run-label line and the column-header line.
 * @param path File path.
 * @return Heap-allocated CsvFile, or NULL on error.
 */
static CsvFile *load_csv(const char *path)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "  ERROR : Cannot open file: %s\n", path);
        return NULL;
    }

    CsvFile *cf   = (CsvFile *)calloc(1, sizeof(CsvFile));
    cf->capacity  = 256;
    cf->rows      = (CsvRow *)malloc((size_t)cf->capacity * sizeof(CsvRow));

    char *line  = (char *)malloc(LINE_BUF);
    char *fbuf  = (char *)malloc(FIELD_BUF);

    /* Line 1: run label */
    if (fgets(line, LINE_BUF, fp)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        strncpy(cf->label, line, sizeof(cf->label) - 1);
    }

    /* Line 2: column header – discard */
    if (fgets(line, LINE_BUF, fp)) { (void)line; }

    /* Data rows */
    while (fgets(line, LINE_BUF, fp)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;

        if (cf->count == cf->capacity) {
            cf->capacity *= 2;
            cf->rows = (CsvRow *)realloc(cf->rows,
                           (size_t)cf->capacity * sizeof(CsvRow));
        }

        CsvRow *row = &cf->rows[cf->count];
        const char *p = line;

        /* col 0 : Query File */
        p = parse_field(p, fbuf, FIELD_BUF);  row->query_file    = strdup(fbuf);
        /* col 1 : Query Length – skip */
        p = parse_field(p, fbuf, FIELD_BUF);
        /* col 2 : Reference File */
        p = parse_field(p, fbuf, FIELD_BUF);  row->ref_file      = strdup(fbuf);
        /* col 3 : Reference Length – skip */
        p = parse_field(p, fbuf, FIELD_BUF);
        /* col 4 : Number of Hits */
        p = parse_field(p, fbuf, FIELD_BUF);  row->num_hits      = atoi(fbuf);
        /* col 5 : Hit Indexes */
        p = parse_field(p, fbuf, FIELD_BUF);  row->hit_indexes   = strdup(fbuf);
        /* col 6 : Lowest Score */
        p = parse_field(p, fbuf, FIELD_BUF);  row->lowest_score  = strdup(fbuf);
        /* col 7 : Lowest Score Indexes */
        p = parse_field(p, fbuf, FIELD_BUF);  row->lowest_indexes= strdup(fbuf);
        /* col 8 : Last Score */
        p = parse_field(p, fbuf, FIELD_BUF);  row->last_score    = atoi(fbuf);

        cf->count++;
    }

    free(line);
    free(fbuf);
    fclose(fp);
    return cf;
}

static void free_csv(CsvFile *cf)
{
    for (int i = 0; i < cf->count; i++) {
        free(cf->rows[i].query_file);
        free(cf->rows[i].ref_file);
        free(cf->rows[i].hit_indexes);
        free(cf->rows[i].lowest_score);
        free(cf->rows[i].lowest_indexes);
    }
    free(cf->rows);
    free(cf);
}

/* ============================================================================
 * INDEX LIST COMPARISON  (order-independent)
 * ============================================================================ */

static int cmp_int_val(const void *a, const void *b)
{
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return (ia > ib) - (ia < ib);
}

/**
 * @brief Compare two comma-separated index lists as unordered sets.
 * @return 1 if the sets are identical, 0 if they differ.
 */
static int index_lists_equal(const char *a, const char *b)
{
    /* Fast path – identical strings */
    if (strcmp(a, b) == 0) return 1;

    int a_na = (strcmp(a, "N/A") == 0 || strlen(a) == 0);
    int b_na = (strcmp(b, "N/A") == 0 || strlen(b) == 0);
    if (a_na && b_na) return 1;
    if (a_na != b_na) return 0;

    /* Count commas to size arrays */
    int cap_a = 1, cap_b = 1;
    for (const char *p = a; *p; p++) if (*p == ',') cap_a++;
    for (const char *p = b; *p; p++) if (*p == ',') cap_b++;

    int *arr_a = (int *)malloc((size_t)cap_a * sizeof(int));
    int *arr_b = (int *)malloc((size_t)cap_b * sizeof(int));
    int  n_a   = 0, n_b = 0;

    /* Parse a */
    char *buf = strdup(a);
    char *tok = strtok(buf, ",");
    while (tok) { arr_a[n_a++] = atoi(tok); tok = strtok(NULL, ","); }
    free(buf);

    /* Parse b */
    buf = strdup(b);
    tok = strtok(buf, ",");
    while (tok) { arr_b[n_b++] = atoi(tok); tok = strtok(NULL, ","); }
    free(buf);

    int equal = 0;
    if (n_a == n_b) {
        qsort(arr_a, (size_t)n_a, sizeof(int), cmp_int_val);
        qsort(arr_b, (size_t)n_b, sizeof(int), cmp_int_val);
        equal = (memcmp(arr_a, arr_b, (size_t)n_a * sizeof(int)) == 0);
    }

    free(arr_a);
    free(arr_b);
    return equal;
}

/* ============================================================================
 * ROW LOOKUP
 * ============================================================================ */

/** Find a row in @p cf by (query_file, ref_file) key. Returns NULL if absent. */
static CsvRow *find_row(CsvFile *cf, const char *qfile, const char *rfile)
{
    for (int i = 0; i < cf->count; i++) {
        if (strcmp(cf->rows[i].query_file, qfile) == 0 &&
            strcmp(cf->rows[i].ref_file,   rfile) == 0)
            return &cf->rows[i];
    }
    return NULL;
}

/* ============================================================================
 * FIELD-LEVEL COMPARISON
 * ============================================================================ */

static FieldMismatch compare_rows(const CsvRow *c, const CsvRow *o)
{
    FieldMismatch fm = {0};
    fm.num_hits       = (c->num_hits != o->num_hits);
    fm.hit_indexes    = !index_lists_equal(c->hit_indexes,    o->hit_indexes);
    fm.lowest_score   = (strcmp(c->lowest_score,  o->lowest_score) != 0);
    fm.lowest_indexes = !index_lists_equal(c->lowest_indexes, o->lowest_indexes);
    fm.last_score     = (c->last_score != o->last_score);
    return fm;
}

static int any_mismatch(const FieldMismatch *fm)
{
    return fm->missing || fm->num_hits || fm->hit_indexes ||
           fm->lowest_score || fm->lowest_indexes || fm->last_score;
}

/* ============================================================================
 * REPORT PRINTING
 * ============================================================================ */

#define SEP_MAJOR \
    "================================================================================\n"
#define SEP_MINOR \
    "--------------------------------------------------------------------------------\n"
#define SEP_THIN  \
    "  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n"

/** Truncate a potentially long string to @p max visible characters for display. */
static void truncate_str(const char *src, char *dst, int max)
{
    int len = (int)strlen(src);
    if (len <= max) {
        strcpy(dst, src);
    } else {
        strncpy(dst, src, (size_t)(max - 3));
        dst[max - 3] = '\0';
        strcat(dst, "...");
    }
}

/**
 * @brief Print one row of the per-pair field comparison table.
 * @param field     Column name.
 * @param val_c     Value from the C reference.
 * @param val_other Value from the compared implementation.
 * @param mismatch  Non-zero if values differ.
 */
static void print_field_row(const char *field,
                             const char *val_c, const char *val_other,
                             int mismatch)
{
    const char *status = mismatch ? "!=" : "==";
    printf("    %-26s  %s  %-38s  %s\n",
           field, status, val_c, val_other);
}

/**
 * @brief Run one full comparison (C vs a single implementation) and print
 *        a formal thesis-style report section.
 *
 * @param c_file     Loaded C reference CSV.
 * @param other_file Loaded implementation CSV.
 * @param other_name Human-readable name for the implementation.
 * @param comp_num   Ordinal of this comparison (1, 2, 3).
 * @param total_comps Total number of comparisons being run.
 */
static void run_comparison(CsvFile *c_file, CsvFile *other_file,
                            const char *other_name,
                            int comp_num, int total_comps)
{
    printf(SEP_MAJOR);
    printf("  Comparison %d of %d : C (CPU Reference)  vs  %s\n",
           comp_num, total_comps, other_name);
    printf(SEP_MINOR);

    int total   = c_file->count;
    int matched = 0, missing = 0;

    MismatchEntry *mismatches =
        (MismatchEntry *)malloc((size_t)total * sizeof(MismatchEntry));
    int mismatch_count = 0;

    for (int i = 0; i < total; i++) {
        CsvRow *crow = &c_file->rows[i];
        CsvRow *orow = find_row(other_file, crow->query_file, crow->ref_file);

        if (!orow) {
            mismatches[mismatch_count].crow      = crow;
            mismatches[mismatch_count].orow      = NULL;
            mismatches[mismatch_count].fm        = (FieldMismatch){0};
            mismatches[mismatch_count].fm.missing = 1;
            mismatch_count++;
            missing++;
            continue;
        }

        FieldMismatch fm = compare_rows(crow, orow);
        if (any_mismatch(&fm)) {
            mismatches[mismatch_count].crow = crow;
            mismatches[mismatch_count].orow = orow;
            mismatches[mismatch_count].fm   = fm;
            mismatch_count++;
        } else {
            matched++;
        }
    }

    int mismatched = mismatch_count - missing;

    /* ── Summary statistics ── */
    printf("  Total pairs evaluated   : %d\n",   total);
    printf("  Matching pairs          : %d\n",   matched);
    printf("  Mismatching pairs       : %d\n",   mismatched);
    printf("  Missing pairs           : %d\n",   missing);
    printf("  Match rate              : %.2f%%\n",
           total > 0 ? (matched * 100.0 / total) : 0.0);
    printf("\n");

    /* ── Verdict ── */
    if (mismatch_count == 0) {
        printf("  VERDICT : PASS — All %d pair(s) produce identical results "
               "to the C reference.\n", total);
    } else {
        printf("  VERDICT : FAIL — %d of %d pair(s) differ from the "
               "C reference.\n", mismatch_count, total);
    }

    /* ── Detailed mismatch report ── */
    if (mismatch_count > 0) {
        printf("\n");
        printf("  Detailed Mismatch Report  (%s  vs  %s)\n",
               c_file->label, other_name);
        printf(SEP_THIN);

        /* Column header for field table */
        printf("    %-26s  %-2s  %-38s  %s\n",
               "Field", "  ", "C Reference", other_name);
        printf("    %-26s  %-2s  %-38s  %s\n",
               "──────────────────────────", "──",
               "──────────────────────────────────────",
               "──────────────────────────────────────");

        for (int m = 0; m < mismatch_count; m++) {
            CsvRow       *crow = mismatches[m].crow;
            CsvRow       *orow = mismatches[m].orow;
            FieldMismatch *fm  = &mismatches[m].fm;

            printf("\n");
            printf("  Pair [%d / %d]  :  %s   vs   %s\n",
                   m + 1, mismatch_count,
                   crow->query_file, crow->ref_file);

            if (fm->missing) {
                printf("    >> This pair was NOT FOUND in the %s results file.\n",
                       other_name);
            } else {
                char c_disp[48], o_disp[48];
                char c_buf[32],  o_buf[32];

                /* Number of Hits */
                snprintf(c_buf, sizeof(c_buf), "%d", crow->num_hits);
                snprintf(o_buf, sizeof(o_buf), "%d", orow->num_hits);
                print_field_row("Number of Hits", c_buf, o_buf,
                                fm->num_hits);

                /* Hit Indexes */
                truncate_str(crow->hit_indexes, c_disp, 45);
                truncate_str(orow->hit_indexes, o_disp, 45);
                print_field_row("Hit Indexes", c_disp, o_disp,
                                fm->hit_indexes);

                /* Lowest Score */
                print_field_row("Lowest Score",
                                crow->lowest_score, orow->lowest_score,
                                fm->lowest_score);

                /* Lowest Score Indexes */
                truncate_str(crow->lowest_indexes, c_disp, 45);
                truncate_str(orow->lowest_indexes, o_disp, 45);
                print_field_row("Lowest Score Indexes", c_disp, o_disp,
                                fm->lowest_indexes);

                /* Last Score */
                snprintf(c_buf, sizeof(c_buf), "%d", crow->last_score);
                snprintf(o_buf, sizeof(o_buf), "%d", orow->last_score);
                print_field_row("Last Score", c_buf, o_buf,
                                fm->last_score);
            }
        }

        printf("\n");
        printf(SEP_THIN);
    }

    free(mismatches);
    printf("\n");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void)
{
    printf("\n");
    printf(SEP_MAJOR);
    printf("  CORRECTNESS VERIFICATION REPORT\n");
    printf("  Algorithm  : Hyyrö Bit-Parallel Damerau-Levenshtein Distance\n");
    printf("  Basis      : C (CPU Reference Implementation)\n");
    printf("  Comparisons: (1) C vs Homo FPGA   "
                           "(2) C vs Homo GPU   "
                           "(3) C vs Hetero\n");
    printf(SEP_MAJOR);
    printf("\n");

    /* ── Load all four CSV files ── */
    printf("  Loading result files...\n");
    printf(SEP_MINOR);

    CsvFile *c_file      = load_csv(CSV_C);
    CsvFile *fpga_file   = load_csv(CSV_FPGA);
    CsvFile *gpu_file    = load_csv(CSV_GPU);
    CsvFile *hetero_file = load_csv(CSV_HETERO);

    if (!c_file || !fpga_file || !gpu_file || !hetero_file) {
        fprintf(stderr,
                "\n  ERROR : One or more CSV files could not be opened. "
                "Aborting.\n\n");
        return EXIT_FAILURE;
    }

    printf("  %-20s : %5d rows   [%s]\n",
           "C Reference",  c_file->count,      c_file->label);
    printf("  %-20s : %5d rows   [%s]\n",
           "Homo FPGA",    fpga_file->count,   fpga_file->label);
    printf("  %-20s : %5d rows   [%s]\n",
           "Homo GPU",     gpu_file->count,    gpu_file->label);
    printf("  %-20s : %5d rows   [%s]\n",
           "Hetero",       hetero_file->count, hetero_file->label);
    printf("\n");

    /* ── Run the three comparisons ── */
    run_comparison(c_file, fpga_file,   "Homo FPGA",  1, 3);
    run_comparison(c_file, gpu_file,    "Homo GPU",   2, 3);
    run_comparison(c_file, hetero_file, "Hetero",     3, 3);

    /* ── Final overall verdict ── */
    printf(SEP_MAJOR);
    printf("  END OF CORRECTNESS VERIFICATION REPORT\n");
    printf(SEP_MAJOR);
    printf("\n");

    free_csv(c_file);
    free_csv(fpga_file);
    free_csv(gpu_file);
    free_csv(hetero_file);

    return EXIT_SUCCESS;
}
