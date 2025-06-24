from src.vectorstore import init_index, client, INDEX_NAME
import argparse
from src.query import run_query
from src.ingest import ingest_documents

def main():
    print("main.py executing under: ", __name__)
    print("__________________________________________________")

    p = argparse.ArgumentParser()
    p.add_argument("--ingest", action ="store_true", help = "Load all docs into Pinecone")
    p.add_argument("-q", "--query", help = "Question to ask the index")
    p.add_argument("--top_k", type = int, default = 10, help = "Number of top results to return for the query")
    p.add_argument("--reset", action = "store_true", help = "Reset the index before ingesting documents")
    args = p.parse_args()

    if not args.ingest and not args.query and not args.reset:
        p.print_help()
        return

    if args.reset:
        print("Resetting the index...")
        if INDEX_NAME in client.list_indexes():
            client.delete_index(INDEX_NAME)
            print("Index reset successfully.")
        else:
            print("Index does not exist, nothing to reset.")

    index = init_index()

    if args.ingest:
        print("Ingesting documents into index")
        ingest_documents(index)
        print("Documents ingested successfully.")

    if args.query:
        print("Querying index with provided query")
        history = []
        user_input = args.query
        while user_input.lower() not in {"exit", "quit"}:
            answer = run_query(index, user_input, top_k=args.top_k, history=history)
            if answer:
                history.append((user_input, answer))
                print(f"\nAssistant: {answer}\n")
            user_input = input("Follow-up (or 'exit'): ").strip()

if __name__ == "__main__":
    main()