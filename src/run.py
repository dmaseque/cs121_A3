import indexer
import merger
import search

def run():
    # indexer.create_inverted_indexes('developer/DEV')
    # merger.merge_partial_indexes()
    search.get_query()

if __name__ == "__main__":
    run()