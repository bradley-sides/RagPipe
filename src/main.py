from src.loader import load_pdf
from src.chunker import chunk_documents
from src.embedder import embed_documents, embed_query
from src.vectorstore import init_index, upsert_chunks, query_index, client, INDEX_NAME
from src.rag import build_prompt, generate_answer
from src.config import DOCS
from src.utils import clean_pages
from src.search_utils  import optimize_query, rerank_chunks
import argparse

def ingest_documents(index):
    for doc_meta in DOCS:
        raw_pages = load_pdf(doc_meta["file_path"])
        cleaned_pages = clean_pages(raw_pages)
        chunks = chunk_documents(cleaned_pages)
        texts = [c.page_content for c in chunks]
        embeddings = embed_documents(texts)
        upsert_chunks(index, 
                      chunks, 
                      embeddings, 
                      base_meta = doc_meta
                      )

def run_query(index, user_query, top_k = 10):
    optimized_q = optimize_query(user_query)
    print(f"Optimized query: {optimized_q}\n")

    results = query_index(
        index,
        embed_query(optimized_q),
        top_k=top_k * 4
    )
    matches = results.get("matches", [])
    if not matches:
        print("No matches found for the query.")
        return  
    
    print(f"Retrieved {len(matches)} candidates, now reranking...\n")
    matches = rerank_chunks(matches, user_query, top_k=top_k)

    display_keys = [
        "company", "fiscal_year", "quarter",
        "call_date", "page", "total_pages",
        "doc_id", "source"
    ]
    print("__________________________________________________")
    for i, m in enumerate(matches, 1):
        md = m["metadata"]
        print(f"--- Match #{i} (score {m['score']:.4f}) ---")
        print("Metadata:")
        for key in display_keys:
            if key in md:
                print(f"  {key}: {md[key]}")
        #print("\nText:")
        #print(md.get("text", "[no text]"))
        print("\n")
    print("__________________________________________________")
    
    chunks_with_meta = []
    for match in matches:
        md = match["metadata"]
        label = f"[{md['company']} | Q{md['quarter']} FY{int(md['fiscal_year'])} • {md['call_date']}]"
        text = md.get("text", "")
        chunks_with_meta.append(f"{label}\n{text}")

    print("Generating answer from top texts...")
    prompt = build_prompt(chunks_with_meta, user_query)
    print(prompt)
    answer = generate_answer(prompt)
    
    print("Answer:\n", answer)

def main():
    print("main.py executing under: ", __name__)
    print("__________________________________________________")
    
    p = argparse.ArgumentParser()
    p.add_argument("--ingest", action = "store_true", help = "Load all docs into Pinecone")
    p.add_argument("-q","--query", help ="Question to ask the index")
    p.add_argument("--top_k", type = int, default = 5, help = "Number of top results to return for the query")
    p.add_argument("--reset", action = "store_true", help ="Reset the index before ingesting documents")
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
        run_query(index, args.query)
        return
    

if __name__ == "__main__":
    main()