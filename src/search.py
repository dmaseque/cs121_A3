import sys
import json
from indexer import inverted_index, tokenize, doc_id_map

# loads entire final_index.json
def load_inverted_index(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")

# search function
# input is the query string
def search(query):
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
        postings = []
        if token in inverted_index:
            postings = inverted_index[token]     

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

    # sort result by tf-idf
    result.sort(key=lambda x: x.get("tf-idf score", 0), reverse=True)

    # doc_id_map is mapping of url -> docIDS
    # reverse to get docIDS -> url
    reversed_doc_id_map = {url: docID for docID, url in doc_id_map.items()}
    
    # return list of docIDs sorted by tf-idf
    return [reversed_doc_id_map[doc["document_id"]] for doc in result]

def get_query():
    # python3 search.py {write search query here}
    # for src/TEST, command line argument = python3 search.py 6pm 
    if len(sys.argv) < 2:
        print("Invalid query.")
        sys.exit(1)

    # form the query string from command line arguments
    query = " ".join(sys.argv[1:])

    print(f"Search query: '{query}'\n")

    inverted_index = load_inverted_index("final_index.json")

    if inverted_index:
        search_result = search(query)
        if search_result:
            print("Documents found from query:")
            for document in search_result:
                print(document)
        else:
            # if search_result is empty, then no documents found in inverted index
            print("No documents found.")
    else:
        print("Inverted Index is empty.")


# if __name__ == '__main__':
    
#     # python3 search.py {write search query here}
#     # for src/TEST, command line argument = python3 search.py 6pm 
#     if len(sys.argv) < 2:
#         print("Invalid query.")
#         sys.exit(1)

#     # form the query string from command line arguments
#     query = " ".join(sys.argv[1:])

#     print(f"Search query: '{query}'\n")

#     inverted_index = load_inverted_index("final_index.json")

#     if inverted_index:
#         search_result = search(query)
#         if search_result:
#             print("Documents found from query:")
#             for document in search_result:
#                 print(document)
#         else:
#             # if search_result is empty, then no documents found in inverted index
#             print("No documents found.")
#     else:
#         print("Inverted Index is empty.")