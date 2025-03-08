# UCI Search Engine

## Requirements
The requirements and dependencies required to run this search engine are listed in `requirements.txt`.
```
beautifulsoup4
nltk
lxml
simhash
scikit-learn
```

## Corpus
The `developer.zip` contains a large collection of ICS web pages. This folder makes up the corpus of the UCI Search Engine.

To access the corpus of webpages, navigate to the `/src` folder.
```
cd src
```
Unzip this folder into the src directory.

[Corpus - large collection of ICS web pages](https://www.ics.uci.edu/~algol/teaching/informatics141cs121w2022/a3files/developer.zip)

## Index Creation
To create the inverted index, run this command line in the terminal:
```
python3 indexer.py
```
Next, merge the partial indexes by running this command line in the terminal:
```
python3 merger.py
```

## Search
To begin search, run this command line in the terminal:
```
python3 run.py
```
The search engine will prompt the user to enter a search query.

To quit, use the command `exit`.
