import indexer
import merger
import search

def run():
    # indexer.create_inverted_indexes('TEST')
    indexer.create_inverted_indexes('/home/cathyw8/cs121_A3/src/TEST')
    merger.merge_partial_indexes()
    search.get_query()

if __name__ == "__main__":
    run()