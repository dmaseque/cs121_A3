import sys
import json
import time
from indexer import inverted_index, tokenize
from sklearn.metrics.pairwise import cosine_similarity
import math

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
    # retrieve total number of documents (N) from bookkeeping file
    total_docs = bookkeeping.get("total_docs", len(doc_id_map)) # resort to len(doc_id_map) if "total docs" not found in postings
     
    # tokenize the query string
    # output is list of (stemmed token, weight)
    query_tokens_weight = tokenize(query, weight=1)
    # get only the stemmed tokens => token[0] (first value of token)
    query_stemmed_tokens = []
    for token in query_tokens_weight:
        query_stemmed_tokens.append(token[0])

    # Initialize result as None, no documents
    result = None

    # for token in query of stemmed tokens, check for token in inverted index
    for token in query_stemmed_tokens:
        # if token in inverted index, retrieve all the postings for that token
        postings = get_postings(token) 

        # AND operation to get intersection of sets

        # if there is nothing in result, add set of docIDs for token in query
        if result is None:
            result = postings
        # if result contains docIDs, only add to result if docIDs in postings AND result
        else:
            # modified from partB of assignment 1

            #sort the result postings and postings by docID
            result.sort(key=lambda x: x.get("document_id"))
            postings.sort(key=lambda x: x.get("document_id"))

            #Algorithm 3: Sorted Lists Approach from Discussion Week 2 slides
            R = []
            result_index = 0
            postings_index = 0
            #iterate through both text files, stop when reached end of one of the files
            while result_index < len(result) and postings_index < len(postings):
                #if file_1 value < file2_value, increment file1_index
                if result[result_index]["document_id"] < postings[postings_index]["document_id"]:
                    result_index += 1
                #if file_2 value < file1_value, increment file2_index
                elif postings[postings_index]["document_id"] < result[result_index]["document_id"]:
                    postings_index += 1
                #if file1_value = file2_value, append to R and increment both indexes
                else:
                    R.append(result[result_index])
                    result_index += 1
                    postings_index += 1
            
            result = R

    # if result is empty, then no documents found in inverted index
    if result == None:
        return []

    # create query vector, weighting with TF-IDF
    query_vector = []
    for token in query_stemmed_tokens:
        # get token's IDF from inverted index
        postings = get_postings(token)

        # IDF used as query term's weight
        # set TF for each query term as 1
        df_t = len(postings)
        idf = math.log(total_docs / (df_t))
        query_vector.append(idf)

     # compute cosine similarity for each document
    doc_similarities = []
    for doc in result:
        doc_id = doc["document_id"]
        doc_vector = []
        for token in query_stemmed_tokens:
            # retrieve postings for the token to get TF-IDF score for the document
            postings = get_postings(token)
            # get TF-IDF score for token ind ocument
            tf_idf = next((posting["tf-idf score"] for posting in postings
                            if posting["document_id"] == doc_id), 0)
            doc_vector.append(tf_idf)

        # use scikit-learn --> compute cosine similarity 
        similarity = cosine_similarity([query_vector], [doc_vector])[0][0]
        doc_similarities.append((doc_id, similarity))

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
