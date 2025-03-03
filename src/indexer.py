import os
import sys
import json 
import re
import nltk
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from simhash import Simhash
from urllib.parse import urlparse
import heapq
import shutil

inverted_index = {} # global variable of inverted index - key: token -> list of postings
index_counter = 1 # current number of index being built

doc_id_map = {}  # document_name (URL) -> mappings of document_id
doc_id_counter = 0  # counter to assign IDs

simhash_set = set() # unique Simhashes for detecting duplicates/near duplicates

"""
MAX_DOCS MUST CHANGE BASED ON DEV (~10000) OR TEST (2)
"""
MAX_DOCS = 10000 # number of documents until it is time to dump

"""
HAMMING_DISTANCE can be modifed by dev for likeness between pages. DEFAULT: 2
"""
HAMMING_DISTANCE = 2 # Distance between simhashes to determine uniqueness

# # download nltk data for tokenization
# nltk.download('punkt')

# deletes the directory at the given path as well as all its contents
def delete_dir(path):
    try:
        shutil.rmtree(path)
    except Exception as e:
        print(f"Error attempting to delete_dir: {e}")

# returns True if url_name doesn't contain invalid extension i.e. .png/.pptx
def is_valid_extension(url):
    try:
        parsed = urlparse(url)
        if re.match(
            r".*\.(css|js|bmp|gif|jpe?g|ico"
            + r"|png|tiff?|mid|mp2|mp3|mp4"
            + r"|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
            + r"|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names"
            + r"|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
            + r"|epub|dll|cnf|tgz|sha1"
            + r"|thmx|mso|arff|rtf|jar|csv"
            + r"|rm|smil|wmv|swf|wma|zip|rar|gz"
            + r"|bak|sql|mdb|db|sqlite|ini|log|cfg"
            + r"|vdi|vmdk|qcow2|img|mat|sav|dta|spss"
            + r"|xz|lzma|zst|tar\.xz|bat|cmd|scr|vbs|apk)$", parsed.path.lower()):
            return False
            
        return True
    except Exception:
        return False

# returns True if tokens of said page belong to a unique Simhash
def is_unique_page(tokens):
    current_hash = Simhash(tokens).value
    for old_hash in simhash_set: # Validates current hash against every hash already encountered
        if bin(current_hash ^ old_hash).count('1') <= HAMMING_DISTANCE:  # Lower hamming_distance = docs must be closer to identical
            return False
    simhash_set.add(current_hash)
    return True
    

# modified tokenize from Part A
def tokenize(text, weight=1):

    # use regular expression to tokenize alphanumeric words in text
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())

    # use Porter stemming for better textual matches
    stemmer = PorterStemmer()
    tokens_stemmed = []     # unigrams
    for token in tokens:
        # only add tokens that are more than 2 characters
        # do not include single letter tokens from contractions
        if len(token) > 2:
            tokens_stemmed.append((stemmer.stem(token), weight))

    # weigh trigram matches higher than bigrams, bigrams than unigrams 
    bigram_weight = 1.25
    trigram_weight = 1.5

    # add 2-grams
    # grabs stemmed word from current and next token, average individual words' weights for bigram's weight
    bigrams = [(f"{tokens_stemmed[i][0]}_{tokens_stemmed[i+1][0]}", 
                ((tokens_stemmed[i][1] + tokens_stemmed[i+1][1]) / 2) * bigram_weight)
                for i in range(len(tokens_stemmed) - 1)]

    # add 3-grams
    trigrams = [(f"{tokens_stemmed[i][0]}_{tokens_stemmed[i+1][0]}_{tokens_stemmed[i+2][0]}", 
                ((tokens_stemmed[i][1] + tokens_stemmed[i+1][1] + tokens_stemmed[i+2][1]) / 2) * trigram_weight)
                for i in range(len(tokens_stemmed) - 2)]
    
    return tokens_stemmed + bigrams + trigrams
    
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
def posting(document_id, term_freq):
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
            "tf": frequency # only store TF during initial indexing phase
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

    # variables for testing
    detected_dups = 0
    detected_bad_extensions = 0

    # delete partial_indexes folder before running to reset
    delete_dir("partial_indexes")

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

            # posting - document_name is the url in the json file
            document_name = content['url']
            document_id = get_document_id(document_name)

            # posting - term frequency score
            
            # skip page if path extension contains invalid extensions i.e. .jpg/.pptx
            if not is_valid_extension(document_name):
                #error_log(f"{document_name} contains an invalid extension", "bad_ext_log")
                detected_bad_extensions += 1
                continue

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

            # determine uniqueness of page by comparing current Simhash against existing Simhashes
            if not is_unique_page(tokens):
                #error_log(f"{document_name} is a duplicate", "dup_log")
                detected_dups += 1
                continue

            term_freq = computeWordFrequencies(tokens)

            # create posting for webpage and add to inverted_index
            posting(document_id, term_freq)

            # DUMP EVERY MAX_DOCS DOCS
            doc_count += 1
            if doc_count % MAX_DOCS == 0:
                dump_inverted_index()
        
    # final dump for remaining memory
    if inverted_index:
        dump_inverted_index()

    # dump mapping
    with open("doc_id_mapping.json", "w", encoding="utf-8") as file:
        json.dump(doc_id_map, file, indent=4)


# save index to json file
def dump_inverted_index():
    global inverted_index, index_counter

    # make a folder called "indexes" if it doesn't exist
    index_folder = "partial_indexes"
    os.makedirs(index_folder, exist_ok=True)

    # name file based on which index it is current on
    output_file = f"partial_indexes/partial_index_{index_counter}.json"
    with open(output_file, "w", encoding="utf-8") as file:
        # sort alphabetically
        json.dump(dict(sorted(inverted_index.items())), file, indent=4)

    # clear memory and increment counter
    inverted_index = {}
    index_counter += 1



def get_document_id(document_name):
    global doc_id_counter, doc_id_map
    if document_name not in doc_id_map:
        doc_id_map[document_name] = doc_id_counter
        doc_id_counter += 1
    return doc_id_map[document_name]

def error_log(msg, path):
    with open(path, 'a') as file:
        file.write(msg + '\n')
    

if __name__ == '__main__':

    # # the DEV folder - extract developer.zip inside the src folder
    create_inverted_indexes('DEV')