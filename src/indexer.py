import os
import sys
import json 
import re
import nltk
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
import heapq

inverted_index = {} # global variable of inverted index - key: token -> list of postings
index_counter = 1 # current number of index being built

"""
MAX_DOCS MUST CHANGE BASED ON DEV (~10000) OR TEST (2)
"""
MAX_DOCS = 10000 # number of documents until it is time to dump

# # download nltk data for tokenization
# nltk.download('punkt')

# modified tokenize from Part A
def tokenize(text, weight=1):

    # use regular expression to tokenize alphanumeric words in text
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())

    # use Porter stemming for better textual matches
    stemmer = PorterStemmer()
    tokens_stemmed = []
    for token in tokens:
        # only add tokens that are more than 2 characters
        # do not include single letter tokens from contractions
        if len(token) > 2:
            tokens_stemmed.append((stemmer.stem(token), weight))
    
    return tokens_stemmed

# computeWordFrequencies from Part A
# Important text: adds default weight 1 and any extra weight for appearing in title (+2), headings(+1), or as bold/strong (+1)
def computeWordFrequencies(tokens):

    word_frequencies = {}

    #iterate through every token/weight in tokens list
    for token, weight in tokens:
        # word_frequencies.get(token, 0) checks if token exists, using 0 as default frequency if it doesn't exist
        # increment frequency if token exists, otherwise set it to its weight
        word_frequencies[token] = word_frequencies.get(token, 0) + weight
        
    return word_frequencies

# add posting to inverted_index
# posting contains document name/id token was found in and its tf-idf score
def posting(document_id, document_name, term_freq):
    global inverted_index
    # iterate through each token
    for token, frequency in term_freq.items():
        # if the token is not in inverted_index dict, add token as key and map to empty list
        # list will later be populated with webpages including token
        if token not in inverted_index:
            inverted_index[token] = []
        
        # posting : document name/id token was found in and its tf-idf score
        posting = {
            "document_id": document_id,
            "document_name": document_name,
            "tf-idf score": frequency
        }

        # print(posting)

        # add posting to inverted_index under the token it was found in
        inverted_index[token].append(posting)
            

# dev is the developer folder
def create_inverted_indexes(dev):

    # in dev, there is one folder per domain
    # corpus is the list of domains in developer folder
    corpus = os.listdir(dev)
    doc_count = 0

    for domain in corpus:
        print(f'Indexing domain:{domain}')
        # json_files for each domain are in folder dev/{domain}
        json_files = '{}/{}'.format(dev, domain)

        # each JSON file coresponds to one web page
        for webpage in os.listdir(json_files):

            #print(f'Processing webpage:{webpage}')
            # webpage_path is dev/{domain}/{webpage}
            webpage_path = os.path.join(json_files, webpage)
        
            # open the json file and load the contents
            try:
                with open(webpage_path, 'r', encoding = 'utf-8') as file:
                    #print('Loading content of the json file')
                    content = json.load(file)
                    file.close()
            except FileNotFoundError:
                print(f'Json File not found for {webpage}.')
            except IOError:
                print(f'Json File input/output error. {webpage}')

            # posting - document_id is name of the json file
            document_id = webpage
            document_id = document_id.removesuffix(".json")
            # posting - document_name is the url in the json file
            document_name = content['url']

            # posting - term frequency score

            try: 
                # parse through content of json file and tokenize text
                soup = BeautifulSoup(content['content'], 'lxml')

                # deal with broken or missing HTML
                # skip document if there's no valid parsed HTML or no meaningful text content
                if not soup or not soup.get_text(strip=True):
                    print(f"Skipping {webpage} due to missing or broken HTML")
                    continue
            except Exception as e:
                print(f"Error parsing HTML for {webpage}: {e}")
                continue

            tokens = []

            # add weights to "important text" (actual weights can be adjusted later)

            # text in titles - additional weight of 2
            if soup.title: # soup.title directly accesses HTML document's <title> tag
                tokens += tokenize(soup.title.get_text(), weight=2)

            # text in headings - additional weight of 1
            for tag in ['h1', 'h2', 'h3']:
                for element in soup.find_all(tag):
                    tokens += tokenize(element.get_text(), weight=1)

            # text in bold/strong - additional weight of 1
            for tag in ['b', 'strong']: 
                for element in soup.find_all(tag):
                    tokens += tokenize(element.get_text(), weight=1)

            # regular text - default weight of 1
            tokens += tokenize(soup.get_text(), weight=1)

            term_freq = computeWordFrequencies(tokens)

            # create posting for webpage and add to inverted_index
            posting(document_id, document_name, term_freq)

            # DUMP EVERY MAX_DOCS DOCS
            doc_count += 1
            if doc_count % MAX_DOCS == 0:
                dump_inverted_index()
        
    # final dump for remaining memory
    if inverted_index:
        dump_inverted_index()

# save index to json file
def dump_inverted_index():
    global inverted_index, index_counter

    # make a folder called "indexes" if it doesn't exist
    index_folder = "partial_indexes"
    os.makedirs(index_folder, exist_ok=True)

    # name file based on which index it is current on
    output_file = f"partial_indexes/partial_index_{index_counter}.json"
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(inverted_index, file, indent=4)

    # clear memory and increment counter
    inverted_index = {}
    index_counter += 1

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
    index_folder = "partial_indexes"
    output_file = "final_index.json"  # Keep this name so it is compatible with generate_report
    partial_files = [os.path.join(index_folder, f) for f in os.listdir(index_folder) if f.startswith("partial_index_") and f.endswith(".json")]
    
    # Sort each partial index in chunks and save to disk
    sorted_chunk_files = []
    for file in partial_files:
        sorted_chunk_files.extend(chunk_sort_and_save(file))
    
    # Open sorted chunks as iterators for merging
    term_streams = [stream_json_terms(file) for file in sorted_chunk_files]
    
    # Use heapq.merge to efficiently merge sorted terms from all partial indexes
    merged_index = {}
    for term, postings in heapq.merge(*term_streams, key=lambda x: x[0]):
        if term in merged_index:
            merged_index[term].extend(postings)
        else:
            merged_index[term] = postings
    
    # Write the final merged index to disk once
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(merged_index, file, indent=4)
    
    print(f"Merged index saved to {output_file}")


if __name__ == '__main__':

    # # the DEV folder - extract developer.zip inside the src folder
    create_inverted_indexes('src/TEST')
    merge_partial_indexes()
