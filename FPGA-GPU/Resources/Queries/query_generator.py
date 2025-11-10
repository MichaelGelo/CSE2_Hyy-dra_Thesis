import random
import os

def generate_dna_sequence(length: int) -> str:
    """Generate a random DNA sequence of given length."""
    bases = ["A", "C", "G", "T"]
    return "".join(random.choice(bases) for _ in range(length))

def save_files(filename_base: str, queries: list):
    """Save queries to TXT and FASTA formats."""
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_base_path = os.path.join(script_dir, filename_base)

    # Save TXT (just sequences)
    with open(f"{output_base_path}.txt", "w") as f:
        for seq in queries:
            f.write(seq + "\n")

    # Save FASTA (with headers)
    with open(f"{output_base_path}.fasta", "w") as f:
        for i, seq in enumerate(queries, 1):
            f.write(f">query_{i}_len{len(seq)}\n")
            f.write(seq + "\n")

    print(f"✅ Files created: {filename_base}.txt and {filename_base}.fasta")

def single_query():
    """Generate one query and save files."""
    length = int(input("Enter query length (max 256): "))
    filename_base = input("Enter the base name for the output files (without extension): ")
    seq = generate_dna_sequence(length)
    save_files(filename_base, [seq])

def multiple_queries():
    """Generate multiple queries of the SAME length and save files."""
    num_queries = int(input("How many queries do you want? "))
    length = int(input("Enter query length (max 256): "))
    filename_base = input("Enter the base name for the output files (without extension): ")
    queries = [generate_dna_sequence(length) for _ in range(num_queries)]
    save_files(filename_base, queries)

if __name__ == "__main__":
    print("DNA Query Generator (for Hyyrö’s Algorithm Testing)")
    print("1. Single query")
    print("2. Multiple queries")

    choice = input("Choose option (1 or 2): ").strip()

    if choice == "1":
        single_query()
    elif choice == "2":
        multiple_queries()
    else:
        print("❌ Invalid option. Please choose 1 or 2.")
