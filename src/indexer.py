import os
import sys
import json 
import regex as re
import nltk
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

# global variable of inverted index - key: token -> list of postings
inverted_index = dict()

# # download nltk data for tokenization
# nltk.download('punkt')

# modified tokenize from Part A
def tokenize(text):

    # use regular expression to tokenize alphanumeric words in text
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())

    # use Porter stemming for better textual matches
    stemmer = PorterStemmer()
    tokens_stemmed = []
    for token in tokens:
        # only add tokens that are more than 2 characters
        # do not include single letter tokens from contractions
        if len(token) > 2:
            tokens_stemmed.append(stemmer.stem(token))
    
    return tokens_stemmed

# computeWordFrequencies from Part A
def computeWordFrequencies(tokens):

    word_frequencies = {}

    #iterate through every token in tokens list
    for token in tokens: 
        #if token exists, increment frequencies
        if token in word_frequencies:
            word_frequencies[token] += 1
        #if token does not exist, set frequency equal to 1
        else:
            word_frequencies[token] = 1
    
    return word_frequencies

# posting contains document name/id token was found in and its tf-idf score
def posting(document_id, document_name, term_freq):
    global inverted_index
    # iterate through each token
    for token, frequency in term_freq:
        # if the token is not in inverted_index dict, add token as key and map to empty list
        # list will later be populated with webpages including token
        if token not in inverted_index:
            inverted_index[token] = {}
        
        # posting : document name/id token was found in and its tf-idf score
        posting = {
            "document_id": document_id,
            "document_name": document_name,
            "tf-idf score": frequency
        }

        # add posting to inverted_index under the token it was found in
        inverted_index[token].append(posting)
            

# dev is the developer folder
def create_inverted_index(dev):

    # in dev, there is one folder per domain
    # corpus is the list of domains in developer folder
    corpus = os.listdir(dev)

    for domain in corpus:
        # json_files for each domain are in folder dev/{domain}
        json_files = '{}/{}'.format(dev, domain)

        # each JSON file coresponds to one web page
        for webpage in os.listdir(json_files):

            # webpage_path is dev/{domain}/{webpage}
            webpage_path = os.path.join(json_files, webpage)

            # open the json file and load the contents
            try:
                with open(webpage_path, 'r', encoding = 'utf-8') as file:
                    content = json.load(file)
                    file.close()
            except FileNotFoundError:
                print('Json File not found.')
            except IOError:
                print('Json File input/output error.')

            # posting - document_id is name of the json file
            document_id = webpage
            # posting - document_name is the url in the json file
            document_name = content['url']

            # posting - term frequency score

            # parse through content of json file and tokenize text
            soup = BeautifulSoup(content['content'], 'lxml')
            text = soup.get_text()
            tokens = tokenize(text)
            term_freq = computeWordFrequencies(tokens)

            # create posting for webpage
            posting(document_id, document_name, term_freq)

if __name__ == '__main__':

    create_inverted_index('../DEV')