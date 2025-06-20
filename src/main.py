from src.vectorstore import init_index, client, INDEX_NAME
import argparse
from src.query import run_query

'''
    Time to run: 
        GPT 4.1 utils: 1:30 - 2 minutes
        GPT 4.0 utils: 30 - 50 seconds 

    Avg cost to run: 2 cents/query
'''
def main():
    print("main.py executing under: ", __name__)
    print("__________________________________________________")
    
    # flags for ingestion, query, top_k adjustment, resetting data store to 0
    p = argparse.ArgumentParser()
    p.add_argument("--ingest", action = "store_true", help = "Load all docs into Pinecone")
    p.add_argument("-q","--query", help ="Question to ask the index")
    p.add_argument("--top_k", type = int, default = 5, help = "Number of top results to return for the query")
    p.add_argument("--reset", action = "store_true", help ="Reset the index before ingesting documents")
    args = p.parse_args()


    if not args.ingest and not args.query and not args.reset:
        p.print_help()
        return
    
    # Set this flag to reset the index
    if args.reset:
        print("Resetting the index...")
        if INDEX_NAME in client.list_indexes():
            client.delete_index(INDEX_NAME)
            print("Index reset successfully.")
        else:
            print("Index does not exist, nothing to reset.")

    # Set this flag to upload new documents into the vector store.
    index = init_index()
    if args.ingest:
        print("Ingesting documents into index")
        ingest_documents(index)
        print("Documents ingested successfully.")

    # Set this flag to query
    if args.query:
        print("Querying index with provided query")
        run_query(index, args.query)
        return
    

if __name__ == "__main__":
    main()