# test 1 = single reference [length = 2048]
# test 2 = multiple reference [100 references] [length = 2048]
# test 3 = multiple reference [100 references] [vary]


import random

def generate_dna_sequence(length: int) -> str:
    """Generate a random DNA sequence of given length."""
    bases = ["A", "C", "G", "T"]
    return "".join(random.choice(bases) for _ in range(length))

def save_files(filename_base: str, sequences: list, label: str = "reference"):
    """Save sequences to TXT and FASTA formats."""
    # Save TXT (just sequences only)
    with open(f"{filename_base}.txt", "w") as f:
        for seq in sequences:
            f.write(seq + "\n")

    # Save FASTA (with headers)
    with open(f"{filename_base}.fasta", "w") as f:
        for i, seq in enumerate(sequences, 1):
            f.write(f">{label}_{i}_len{len(seq)}\n")
            f.write(seq + "\n")

    print(f"✅ Files created: {filename_base}.txt and {filename_base}.fasta")

# ---------------- Tests ---------------- #

def test1_single_reference():
    """Test 1: Single reference [user-defined length]"""
    length = int(input("Enter reference length: "))
    seq = generate_dna_sequence(length)
    filename_base = f"ref_test1_single-{length}"
    save_files(filename_base, [seq], label="ref")

def test2_multiple_references_fixed():
    """Test 2: Multiple references [user-defined count & length]"""
    num_refs = int(input("Enter number of references: "))
    length = int(input("Enter reference length: "))
    refs = [generate_dna_sequence(length) for _ in range(num_refs)]
    filename_base = f"ref_test2_multi[{num_refs}]-{length}"
    save_files(filename_base, refs, label="ref")

def test3_multiple_references_vary():
    """Test 3: Multiple references [user-defined count & min/max length]"""
    num_refs = int(input("Enter number of references: "))
    min_len = int(input("Enter minimum reference length: "))
    max_len = int(input("Enter maximum reference length: "))

    refs = []
    for _ in range(num_refs):
        length = random.randint(min_len, max_len)
        refs.append(generate_dna_sequence(length))

    filename_base = f"ref_test3_multi[{num_refs}]-vary"
    save_files(filename_base, refs, label="ref")

# ---------------- Main ---------------- #

if __name__ == "__main__":
    print("Reference Generator (for Hyyrö’s Algorithm Testing)")
    print("1. Test 1: Single reference [choose length]")
    print("2. Test 2: Multiple references [choose count & fixed length]")
    print("3. Test 3: Multiple references [choose count & varying length range]")

    choice = input("Choose test (1/2/3): ").strip()

    if choice == "1":
        test1_single_reference()
    elif choice == "2":
        test2_multiple_references_fixed()
    elif choice == "3":
        test3_multiple_references_vary()
    else:
        print("❌ Invalid option. Please choose 1, 2, or 3.")
