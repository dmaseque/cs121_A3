import heapq
import json
import time
from indexer import tokenize
from sklearn.metrics.pairwise import cosine_similarity
import math
import numpy as np

# Load doc to id mapping
with open("doc_id_mapping.json", 'r', encoding='utf-8') as file:
    doc_id_map = json.load(file)
    doc_id_map = {v: k for k, v in doc_id_map.items()}

# Load bookkeeping file to map terms to byte positions in the index file
with open("bookkeeping.json", 'r', encoding='utf-8') as book_file:
    bookkeeping = json.load(book_file)

def get_postings(term):
    if term not in bookkeeping:
        return []
    
    # Get byte offset
    position = bookkeeping[term]

    # Open the index file and seek to the term's position
    with open("final_index.json", "r", encoding="utf-8") as index_file:
        # Move file pointer to term's location
        index_file.seek(position)
        # Read the term's data
        line = index_file.readline().strip().rstrip(',')
        data = json.loads("{" + line + "}") 
        # # only include the first 25% of tf-idf scores, unless is goes below the minimum (25)
        x = int(len(data[term])*.25) if len(data[term]) >= 100 else len(data[term])

        return data[term][:x]

# search function
# input is the query string
def search(query):
    def get_cached_postings(term):
        if term not in postings_cache:
            postings_cache[term] = get_postings(term)
        return postings_cache[term]
    postings_cache = {}
    
    # retrieve total number of documents (N) from bookkeeping file
    total_docs = bookkeeping.get("total_docs", len(doc_id_map)) # resort to len(doc_id_map) if "total docs" not found in postings
     
    # tokenize the query string
    # output is list of (stemmed token, weight)
    query_tokens_weight = tokenize(query, weight=1)
    # get only the stemmed tokens => token[0] (first value of token)
    query_stemmed_tokens = [token[0] for token in query_tokens_weight]

    # Initialize result as None, no documents
    result = None

    # create query vector, weighting with TF-IDF
    query_vector = []
    for token in query_stemmed_tokens:
        # get token's IDF from inverted index
        postings = get_cached_postings(token)

        # IDF used as query term's weight
        # set TF for each query term as 1
        df_t = len(postings)
        # to avoid ZeroDivisionError, handle casse where df_t is 0 (query terms don't exist in any of the indexed documents)
        idf = math.log((total_docs + 1) / (df_t + 1)) + 1  # Smoothed IDF
        query_vector.append(idf)

    # compute cosine similarity for each document
    # Precompute TF-IDF scores for all tokens in query
    tf_idf_lookup = {}  # {document_id: np.array(tf-idf scores)}

    for token in query_stemmed_tokens:
        postings = get_cached_postings(token)  # Retrieve postings once per token

        # Extract document IDs from postings
        postings_ids = {doc["document_id"] for doc in postings}

        # Perform set intersection
        if result is None:
            result = postings_ids  # First token sets the initial result
        else:
            result &= postings_ids  # Keep only common document IDs

        for posting in postings:
            doc_id = posting["document_id"]
            if doc_id in result:  # Only consider intersected documents
                if doc_id not in tf_idf_lookup:
                    tf_idf_lookup[doc_id] = np.zeros(len(query_stemmed_tokens))
                tf_idf_lookup[doc_id][query_stemmed_tokens.index(token)] = posting["tf-idf score"]

    # Convert query vector to numpy array
    query_vector = np.array(query_vector).reshape(1, -1)

    # Compute cosine similarities efficiently
    doc_similarities = [
        (doc_id, cosine_similarity(query_vector, doc_vector.reshape(1, -1))[0][0])
        for doc_id, doc_vector in tf_idf_lookup.items()
    ]

    # sort documents by cosine similarity
    doc_similarities.sort(key=lambda x: x[1], reverse=True)

    # return list of docIDs sorted by cosine similarity
    return [doc_id_map[doc_id] for doc_id, _ in doc_similarities][:5]

def get_query():
    while True:
        # Prompt user for query
        query = input("Enter search query (or type 'exit' to quit): ").strip()
        
        if query.lower() == "exit":
            print("Exiting search.")
            break

        if not query:
            print("Invalid query. Please enter a valid search term.")
            continue

        print(f"\nSearch query: '{query}'\n")

        # Start timer
        start_time = time.time()

        search_result = search(query)
        
        # End timer
        end_time = time.time()
        # Get difference and convert to milliseconds
        elapsed_time = (end_time - start_time) * 1000 

        if search_result:
            print("Documents found from query:")
            for document in search_result:
                print(document)
        else:
            print("No documents found.")

        print(f"Query execution time: {elapsed_time:.2f} ms\n")


if __name__ == '__main__':
    get_query()
