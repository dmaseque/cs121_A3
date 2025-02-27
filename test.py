import json

def get_term_info(term, index_file="final_index.json", bookkeeping_file="bookkeeping.json"):
    # Load the bookkeeping file to find the byte position of the term
    with open(bookkeeping_file, "r", encoding="utf-8") as book_file:
        bookkeeping = json.load(book_file)

    # Check if the term exists in the bookkeeping file
    if term not in bookkeeping:
        print(f"Term '{term}' not found in bookkeeping.")
        return None

    position = bookkeeping[term]  # Get the byte offset of the term

    # Open the index file and seek to the term's position
    with open(index_file, "r", encoding="utf-8") as index_file:
        index_file.seek(position)  # Move file pointer to term's location
        line = index_file.readline().strip().rstrip(',')  # Read the term's data
        data = json.loads("{" + line + "}") 

        documents = data[term]
        print(documents)

# Example Usage
get_term_info("all")