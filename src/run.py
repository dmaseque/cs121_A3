import indexer
import merger

def run():
    indexer.create_inverted_indexes('src/DEV')
    merger.merge_partial_indexes()

if __name__ == "__main__":
    run()