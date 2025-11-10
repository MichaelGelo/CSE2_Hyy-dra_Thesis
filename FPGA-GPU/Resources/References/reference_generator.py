import random
import os

def generate_dna_sequence(length: int) -> str:
    """Generate a random DNA sequence of given length."""
    bases = ["A", "C", "G", "T"]
    return "".join(random.choice(bases) for _ in range(length))

def save_files(filename_base: str, sequences: list, label: str = "reference"):
    """Save sequences to TXT and FASTA formats."""
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_base_path = os.path.join(script_dir, filename_base)

    # Save TXT (just sequences only)
    with open(f"{output_base_path}.txt", "w") as f:
        for seq in sequences:
            f.write(seq + "\n")

    # Save FASTA (with headers)
    with open(f"{output_base_path}.fasta", "w") as f:
        for i, seq in enumerate(sequences, 1):
            f.write(f">{label}_{i}_len{len(seq)}\n")
            f.write(seq + "\n")

    print(f"✅ Files created: {filename_base}.txt and {filename_base}.fasta")

# ---------------- Tests ---------------- #

def test1_single_reference():
    """Test 1: Single reference [user-defined length]"""
    length = int(input("Enter reference length: "))
    filename_base = input("Enter the base name for the output files (without extension): ")
    seq = generate_dna_sequence(length)
    save_files(filename_base, [seq], label="ref")

def test2_multiple_references_fixed():
    """Test 2: Multiple references [user-defined count & fixed length]"""
    num_refs = int(input("Enter number of references: "))
    length = int(input("Enter reference length: "))
    filename_base = input("Enter the base name for the output files (without extension): ")
    refs = [generate_dna_sequence(length) for _ in range(num_refs)]
    save_files(filename_base, refs, label="ref")

def test3_multiple_references_custom_lengths():
    """Test 3: Multiple references [user-defined count & custom length per reference]"""
    num_refs = int(input("Enter number of references: "))
    filename_base = input("Enter the base name for the output files (without extension): ")
    refs = []
    lengths = []

    for i in range(1, num_refs + 1):
        length = int(input(f"Enter length for reference {i}: "))
        lengths.append(length)
        refs.append(generate_dna_sequence(length))

    save_files(filename_base, refs, label="ref")
    print("🧬 Reference lengths:", lengths)

def introduce_edits(sequence: str, num_edits: int) -> str:
    """Introduce a specified number of random edits into a sequence."""
    seq_list = list(sequence)
    bases = ["A", "C", "G", "T"]
    
    for _ in range(num_edits):
        # Ensure there's sequence to edit
        if not seq_list:
            print("⚠️ Warning: Sequence became empty during edits.")
            break
            
        edit_type = random.choice(['insertion', 'deletion', 'substitution', 'transposition'])

        if edit_type == 'insertion':
            pos = random.randint(0, len(seq_list))
            base = random.choice(bases)
            seq_list.insert(pos, base)
        elif edit_type == 'deletion' and len(seq_list) > 0:
            pos = random.randrange(len(seq_list))
            seq_list.pop(pos)
        elif edit_type == 'substitution' and len(seq_list) > 0:
            pos = random.randrange(len(seq_list))
            original_base = seq_list[pos]
            new_bases = [b for b in bases if b != original_base]
            seq_list[pos] = random.choice(new_bases)
        elif edit_type == 'transposition' and len(seq_list) > 1:
            pos = random.randrange(len(seq_list) - 1)
            seq_list[pos], seq_list[pos+1] = seq_list[pos+1], seq_list[pos]
        else: # Handle cases where deletion/substitution/transposition is not possible
            # Default to insertion if other ops fail (e.g., empty list for deletion)
            pos = random.randint(0, len(seq_list))
            base = random.choice(bases)
            seq_list.insert(pos, base)
            
    return "".join(seq_list)

def test4_embedded_edited_query():
    """Test 4: Generate a reference with an embedded, edited query."""
    query = input("Enter the query sequence to embed: ")
    ref_length = int(input("Enter the total length for the reference sequence: "))
    num_edits = int(input("Enter the number of edits (insert/del/sub/transpose) to apply to the query: "))
    filename_base = input("Enter the base name for the output files (without extension): ")

    edited_query = introduce_edits(query, num_edits)
    
    print(f"\nOriginal query ({len(query)}bp): {query}")
    print(f"Edited query   ({len(edited_query)}bp): {edited_query}")

    remaining_len = ref_length - len(edited_query)
    random_dna = generate_dna_sequence(remaining_len)
    insert_pos = random.randint(0, remaining_len)
    final_ref = random_dna[:insert_pos] + edited_query + random_dna[insert_pos:]
    
    save_files(filename_base, [final_ref], label="ref_with_embedded")
    print(f"↪️  Edited query was embedded at position {insert_pos}.")

def test5_multiple_embedded_queries():
    """Test 5: Generate a single reference with MULTIPLE embedded, edited queries."""
    ref_length = int(input("Enter the total length for the final reference sequence: "))
    num_queries_to_embed = int(input("Enter number of queries to embed in the reference: "))
    filename_base = input("Enter the base name for the output files (without extension): ")
    
    edited_queries = []
    total_edited_len = 0
    
    for i in range(1, num_queries_to_embed + 1):
        print(f"\n--- Query {i} ---")
        query = input(f"Enter original sequence for query {i}: ")
        num_edits = int(input(f"Enter number of edits for query {i}: "))

        edited_query = introduce_edits(query, num_edits)
        edited_queries.append(edited_query)
        total_edited_len += len(edited_query)
        
        print(f"Original query ({len(query)}bp): {query}")
        print(f"Edited query   ({len(edited_query)}bp): {edited_query}")

    remaining_len = ref_length - total_edited_len
    if remaining_len < 0:
        print(f"\n❌ Error: Total length of edited queries ({total_edited_len}bp) is greater than the desired reference length ({ref_length}bp).")
        return

    # Shuffle the edited queries to randomize their order in the reference
    random.shuffle(edited_queries)

    # Generate filler DNA and split it to insert queries
    filler_dna = generate_dna_sequence(remaining_len)
    split_points = sorted([random.randint(0, remaining_len) for _ in range(len(edited_queries))])
    
    final_ref_parts = []
    last_pos = 0
    for i, query in enumerate(edited_queries):
        final_ref_parts.append(filler_dna[last_pos:split_points[i]])
        final_ref_parts.append(query)
        last_pos = split_points[i]
    final_ref_parts.append(filler_dna[last_pos:])
    
    final_ref = "".join(final_ref_parts)
    save_files(filename_base, [final_ref], label="ref_with_multi_embed")
    print(f"↪️  {num_queries_to_embed} edited queries were embedded into a single reference of length {len(final_ref)}bp.")

# ---------------- Main ---------------- #

if __name__ == "__main__":
    print("Reference Generator (for Hyyrö’s Algorithm Testing)")
    print("1. Test 1: Single reference [choose length]")
    print("2. Test 2: Multiple references [choose count & fixed length]")
    print("3. Test 3: Multiple references [choose count & custom length per reference]")
    print("4. Test 4: Generate a single reference with one embedded, edited query")
    print("5. Test 5: Generate a single reference with MULTIPLE embedded, edited queries")

    choice = input("Choose test (1/2/3/4/5): ").strip()

    if choice == "1":
        test1_single_reference()
    elif choice == "2":
        test2_multiple_references_fixed()
    elif choice == "3":
        test3_multiple_references_custom_lengths()
    elif choice == "4":
        test4_embedded_edited_query()
    elif choice == "5":
        test5_multiple_embedded_queries()
    else:
        print("❌ Invalid option. Please choose 1, 2, 3, 4, or 5.")
