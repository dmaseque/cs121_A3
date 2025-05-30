import os
import json 
import re
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from simhash import Simhash
from urllib.parse import urlparse, urlsplit, urlunsplit
import shutil
from lxml import html

inverted_index = {} # global variable of inverted index - key: token -> list of postings
index_counter = 1 # current number of index being built

doc_id_map = {}  # document_name (URL) -> (document_id, link to json file)
doc_id_counter = 0  # counter to assign IDs

simhash_set = set() # unique Simhashes for detecting duplicates/near duplicates

"""
MAX_DOCS MUST CHANGE BASED ON DEV (~10000) OR TEST (2)
"""
MAX_DOCS = 8000 # number of documents until it is time to dump

"""
HAMMING_DISTANCE can be modifed by dev for likeness between pages. DEFAULT: 2
"""
HAMMING_DISTANCE = 4 # Distance between simhashes to determine uniqueness

"""
MAX_FILE_SIZE helps prevent indexing large json files, likely to have little valuable info. DEFAULT: 1000KB
"""
MAX_FILE_SIZE = 1000 * 1024  # 1000KB in bytes

# # download nltk data for tokenization
# nltk.download('punkt')

# deletes the directory at the given path as well as all its contents
def delete_dir(path):
    try:
        shutil.rmtree(path)
    except Exception as e:
        print(f"Error attempting to delete_dir: {e}")

# returns True if url_name is valid
def is_valid(url):
    try:

        parsed = urlparse(url)
        file_extensions = ( r".*\.(css|js|bmp|gif|jpe?g|ico|img|png|tiff?|mid|mp2|mp3|mp4|"
                            r"wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf|ps|eps|tex|ppt|pptx|doc|"
                            r"docx|xls|xlsx|names|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso|"
                            r"epub|dll|cnf|tgz|sha1|thmx|mso|arff|rtf|jar|csv|rm|smil|wmv|swf|"
                            r"wma|zip|rar|gz|war|apk|mpg|bam|emx|bib|shar|lif|ppsx|wvx|odc|pps|xml|fig|dtd|sql|java|cp|sh|svg|conf|ipynb|json|scm|ff|py|log|model|cc|sas|tsv|map|DS_Store)$" )
        # Exclude file extensions in params and query
        if re.match(file_extensions, parsed.path.lower()) or re.match(file_extensions, parsed.query.lower()):
            return False
        
        # Avoid txt from these paths since they are all data or code examples
        if re.search(r'(~wjohnson|~babaks|~jacobson|bibtex|~stasio|~kay|~seal).*\.txt$', parsed.path.lower()):
            return False
        
        low_value_patterns = ["raw-attachment", "public_data"]
        if any(pattern in parsed.path.lower() or pattern in parsed.query.lower() for pattern in low_value_patterns):
            return False

        # Return true when all filters are passed
        return True
    
    except TypeError:
        print ("TypeError for ", parsed)
        raise

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
    synonym_map = {
        "crista": "cristina",
        "cs": "compsci"
    }

    # use regular expression to tokenize alphanumeric words in text
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())

    # use Porter stemming for better textual matches
    stemmer = PorterStemmer()
    tokens_stemmed = []     # unigrams
    unique_tokens = set()
    for token in tokens:
        # Normalize using synonym map
        token = synonym_map.get(token, token)
        # only add tokens that are more than 2 characters
        # do not include numbers > 5 digits
        if len(token) > 2 and not (token.isdigit() and len(token) > 5):
            tokens_stemmed.append((stemmer.stem(token), weight))
            unique_tokens.add(token)
    # if there is too much replication, ignore
    if len(tokens) == 0 or len(unique_tokens)/len(tokens) < 0.05:
        return []

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

    # Find the maximum frequency in the word_frequencies
    max_freq = max(word_frequencies.values(), default=1)

    # normalize and scale to the range 0-100
    for token in word_frequencies:
        # Normalize by dividing by max frequency, then scale to 0-100
        word_frequencies[token] = round((word_frequencies[token] / max_freq) * 100, 3)
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
    # delete_dir("partial_indexes")

    for domain in corpus:
        print(f'Indexing domain:{domain}')
        # json_files for each domain are in folder dev/{domain}
        json_files = '{}/{}'.format(dev, domain)

        # each JSON file coresponds to one web page
        for webpage in os.listdir(json_files):

            #print(f'Processing webpage:{webpage}')
            # webpage_path is dev/{domain}/{webpage}
            webpage_path = os.path.join(json_files, webpage)

            # Check file size
            if os.path.getsize(webpage_path) > MAX_FILE_SIZE:
                print(f"Skipping {webpage} due to file size > 1000KB")
                continue
        
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

            parts = urlsplit(document_name)
            document_name = urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ''))

            # posting - term frequency score
            
            # skip page if path extension contains invalid url
            if not is_valid(document_name):
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

            # index on anchor words

            # transform content to bytes
            content_bytes = content["content"]

            if content_bytes.strip():
                if isinstance(content_bytes, str):
                    content_bytes = content_bytes.encode("utf-8")

                try:
                    tree = html.fromstring(content_bytes)
                except Exception as e:
                    print(f"Error parsing HTML for {webpage} aka {document_name}: {e}")
                    continue

                # retrieve all the anchor words in a list anchor_text
                for anchor in tree.xpath("//a[@href]"):
                    anchor_text = anchor.text_content().strip().lower()

                    # turn anchor words into tokens and give a large weight because it contains target url
                    tokens += tokenize(anchor_text, weight=5)

            # add weights to "important text" (actual weights can be adjusted later)
            # text in titles - additional weight of 2
            if soup.title: # soup.title directly accesses HTML document's <title> tag
                tokens += tokenize(soup.title.get_text(), weight=5) # testing adjustment to 5

            # text in headings - additional weight of 1
            for tag in ['h1', 'h2', 'h3']:
                for element in soup.find_all(tag):
                    tokens += tokenize(element.get_text(), weight=3) # testing adjustment to 3

            # text in bold/strong - additional weight of 1
            for tag in ['b', 'strong']: 
                for element in soup.find_all(tag):
                    tokens += tokenize(element.get_text(), weight=2) # testing adjustment to 2

            # regular text - default weight of 1
            tokens += tokenize(soup.get_text(), weight=1)

            # determine uniqueness of page by comparing current Simhash against existing Simhashes
            if not is_unique_page(tokens):
                #error_log(f"{document_name} is a duplicate", "dup_log")
                detected_dups += 1
                continue

            # if no valid tokens, move on
            if tokens == []:
                print(f"Skipping {webpage} due to no valid tokens")
                continue

            term_freq = computeWordFrequencies(tokens)

            # create posting for webpage and add to inverted_index
            document_id = get_document_id(document_name, webpage) # do this here to avoid adding dupes to doc_id_map
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
        json.dump(doc_id_map, file, indent=1)


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
        json.dump(dict(sorted(inverted_index.items())), file, indent=1)

    # clear memory and increment counter
    inverted_index = {}
    index_counter += 1



def get_document_id(document_name, webpage):
    global doc_id_counter, doc_id_map
    if document_name not in doc_id_map:
        doc_id_map[document_name] = (doc_id_counter, webpage)
        doc_id_counter += 1
    return doc_id_map[document_name][0]

def error_log(msg, path):
    with open(path, 'a') as file:
        file.write(msg + '\n')
    

if __name__ == '__main__':

    # # the DEV folder - extract developer.zip inside the src folder
    create_inverted_indexes('DEV')
