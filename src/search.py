import json
import time
from indexer import tokenize, computeWordFrequencies
from sklearn.metrics.pairwise import cosine_similarity
import math
import numpy as np

# postings_cache = {}

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

        return data[term]

# search function
# input is the query string
def search(query):
    def get_cached_postings(term):
        if term not in postings_cache:
            postings_cache[term] = get_postings(term)
        return postings_cache[term]
    postings_cache = {}
    
    # retrieve total number of documents (N) from bookkeeping file
    total_docs = bookkeeping.get("total_docs", len(doc_id_map))  # resort to len(doc_id_map) if "total docs" not found in postings
     
    # tokenize the query string
    # output is list of (stemmed token, weight)
    query_tokens_weight = tokenize(query, weight=1)

    # get only the stemmed tokens => token[0] (first value of token)
    query_stemmed_tokens = [token[0] for token in query_tokens_weight]
    query_freqs = computeWordFrequencies([(token, 1) for token in query_stemmed_tokens])

    # query_tokens = query.lower().split()

    # query_tokens.extend(query_stemmed_tokens)

    # query_tokens = list(set(query_tokens))

    # Initialize result as None, no documents
    result = None

        for token in query_stemmed_tokens:
            if "_" in token:
                continue
            # retrieve postings once per token
            postings = get_cached_postings(token)[:200]

            # extract doc ids from postings
            postings_ids = {doc["document_id"] for doc in postings}

            # set intersection
            if result is None:
                # initial result is first token sets
                result = postings_ids
            else:
                # only keep common doc ids
                result &= postings_ids

        # create query vector, weighted with tf-idf
        query_vector = []

        # compute cosine similarity for each doc, and tf-idf scores for all query tokens
        tf_idf_lookup = {} 
        
        for token in query_stemmed_tokens:
            # retrieve postings once per token
            postings = get_cached_postings(token)

            tf = query_freqs[token]
            df_t = len(postings)
            # to avoid ZeroDivisionError, handle casse where df_t is 0 (query terms don't exist in any of the indexed documents)
            idf = math.log((total_docs + 1) / (df_t + 1))  # smoothed IDF
            tf_idf = (1 + math.log(tf)) * idf
            query_vector.append(round(tf_idf, 3))

            postings = postings[:200]

            for posting in postings:
                doc_id = posting["document_id"]
                # only consider intersected documents
                if doc_id in result: 
                    if doc_id not in tf_idf_lookup:
                        tf_idf_lookup[doc_id] = np.zeros(len(query_stemmed_tokens))
                    tf_idf_lookup[doc_id][query_stemmed_tokens.index(token)] = posting["tf-idf score"]

        # convert query vector--> numpy array
        query_vector = np.array(query_vector).reshape(1, -1)

        # compute cosine similarities
        doc_similarities = [
            (doc_id, cosine_similarity(query_vector, doc_vector.reshape(1, -1))[0][0])
            for doc_id, doc_vector in tf_idf_lookup.items()
        ]

        # sort documents by cosine similarity
        doc_similarities.sort(key=lambda x: x[1], reverse=True)

        # return list of docIDs sorted by cosine similarity
        return [doc_id_map[doc_id] for doc_id, _ in doc_similarities][:5]

    # for phrase queries, rank valid documents by TF-IDF
    if is_phrase_query:
        # compute tf-idf scores for valid documents
        tf_idf_lookup = {}
        for doc_id in result:
            tf_idf_lookup[doc_id] = np.zeros(len(query_stemmed_tokens))
            for i, token in enumerate(query_stemmed_tokens):
                postings = get_cached_postings(token)
                doc_postings = [p for p in postings if p["document_id"] == doc_id]
                if doc_postings:
                    tf_idf_lookup[doc_id][i] = doc_postings[0]["tf-idf score"]

        # convert query vector to numpy array
        query_vector = np.array(query_vector).reshape(1, -1)

        # compute cosine similarities efficiently
        doc_similarities = [
            (doc_id, cosine_similarity(query_vector, doc_vector.reshape(1, -1))[0][0])
            for doc_id, doc_vector in tf_idf_lookup.items()
        ]

        # sort documents by cosine similarity
        doc_similarities.sort(key=lambda x: x[1], reverse=True)

        # return list of docIDs sorted by cosine similarity
        return [doc_id_map[doc_id] for doc_id, _ in doc_similarities][:5]

    return []  # if no results

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
