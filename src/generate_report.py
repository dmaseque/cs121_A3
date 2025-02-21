import json
import os

def generate_report():
    # Load JSON file
    with open("inverted_index.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    # Number of unique tokens (top-level keys)
    num_unique_tokens = len(data.keys())

    # Collect unique document IDs across all tokens
    unique_documents = set()
    for token in data.values():
        for doc in token:
            unique_documents.add(doc["document_id"])

    num_documents = len(unique_documents)

    # File size in KB
    file_size_kb = os.path.getsize("inverted_index.json") / 1024  # Convert bytes to KB

    with open("report.txt", "w", encoding="utf-8") as file:
        file.write(f"Number of Documents: {num_documents}\n")
        file.write(f"Number of Unique Tokens: {num_unique_tokens}\n")
        file.write(f"Total Index Size (KB): {round(file_size_kb, 2)}\n")

if __name__ == "__main__":
    generate_report()