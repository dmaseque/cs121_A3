import json
import os
import heapq

# Sorts small chunks of a large file and writes them back to disk in JSON Lines format
def chunk_sort_and_save(file_path, chunk_size=10000):
    
    temp_files = []
    with open(file_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
        terms = list(index_data.items())
        
        for i in range(0, len(terms), chunk_size):
            chunk = sorted(terms[i:i + chunk_size])  # Sort only a small chunk
            temp_file = f"{file_path}_chunk_{i}.json"
            with open(temp_file, "w", encoding="utf-8") as out_f:
                for term, postings in chunk:
                    json.dump({term: postings}, out_f)
                    out_f.write("\n")  # Each term is now on a separate line
            temp_files.append(temp_file)
    return temp_files  # Return paths to sorted chunks

# Yields terms and postings from a JSON Lines file without loading full file into memory.
def stream_json_terms(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:  # Read one line (one term) at a time
            term_data = json.loads(line.strip())  # Convert JSON string to dictionary
            term, postings = next(iter(term_data.items()))  # Extract term and postings
            yield (term, postings)

def merge_partial_indexes():
    bookkeeper = {}

    bookkeeper_file = "bookkeeping.json"
    index_folder = "partial_indexes"
    output_file = "final_index.json"  # Keep this name so it is compatible with generate_report
    partial_files = [os.path.join(index_folder, f) for f in os.listdir(index_folder) if f.startswith("partial_index_") and f.endswith(".json")]
    
    # Sort each partial index in chunks and save to disk
    sorted_chunk_files = []
    for file in partial_files:
        sorted_chunk_files.extend(chunk_sort_and_save(file))
    
    # Open sorted chunks as iterators for merging
    term_streams = [stream_json_terms(file) for file in sorted_chunk_files]
    
    # Open final index for writing
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("{\n")  # JSON start
        first_entry = True

        current_term = None
        current_postings = []

        for term, postings in heapq.merge(*term_streams, key=lambda x: x[0]):
            if term == current_term:
                current_postings.extend(postings)  # Merge postings
            else:
                # Write the previous term to disk before moving to the next
                if current_term is not None:
                    if not first_entry:
                        file.write(",\n")
                    first_entry = False

                    # Store byte position before writing the term
                    bookkeeper[current_term] = file.tell()

                    current_postings.sort(key=lambda x: x.get("tf-idf score", 0), reverse=True)  

                    # Write term correctly without extra quotes
                    file.write(f'"{current_term}": ')  
                    json.dump(current_postings, file)  

                # Start new term
                current_term = term
                current_postings = postings

        # Write last term
        if current_term is not None:
            if not first_entry:
                file.write(",\n")
            bookkeeper[current_term] = file.tell()
            file.write(f'"{current_term}": ')
            json.dump(current_postings, file)

        file.write("\n}")  # JSON end

    print(f"Merged index saved to {output_file}")

    # Save the secondary index (bookkeeping file)
    with open(bookkeeper_file, "w", encoding="utf-8") as bookkeeper_file:
        json.dump(bookkeeper, bookkeeper_file, indent=4)
    print(f"Bookkeeping file saved to {bookkeeper_file}")

    # Delete chunk files
    for chunk_file in sorted_chunk_files:
        os.remove(chunk_file)

if __name__ == '__main__':

    # # the DEV folder - extract developer.zip inside the src folder
    merge_partial_indexes()