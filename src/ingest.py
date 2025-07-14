from src.loader import load_pdf, load_txt
from src.chunker import chunk_documents
from src.embedder import embed_documents
from src.vectorstore import upsert_chunks
from src.config import DOCS, DOCS_PATH
from src.utils import clean_pages
import json

'''
    Ingest documents into Pinecone vector store.
    If doc_id is provided, only that doc is ingested.
    Otherwise, ingest all docs in DOCS.
'''
def ingest_documents(index, doc_id=None):
    with open(DOCS_PATH) as f:
        docs = json.load(f)

    docs_to_process = docs
    if doc_id and doc_id.lower() != "all":
        print(f"processing: {doc_id}")
        docs_to_process = [
            doc for doc in docs if doc["doc_id"].lower() == doc_id.lower()
        ]
        if not docs_to_process:
            print(f"⚠️ No document found with doc_id: {doc_id}")
            return

    for doc_meta in docs_to_process:
        print(f"📄 Ingesting {doc_meta['doc_id']} ...")

        # 🟢 Load PDF or TXT properly
        if doc_meta["file_path"].endswith(".pdf"):
            raw_pages = load_pdf(doc_meta["file_path"])
        elif doc_meta["file_path"].endswith(".txt"):
            raw_pages = load_txt(doc_meta["file_path"])
        else:
            print(f"⚠️ Unknown file type for: {doc_meta['file_path']}")
            continue

        cleaned_pages = clean_pages(raw_pages)

        chunks = chunk_documents(cleaned_pages)
        texts = [c.page_content for c in chunks]

        print(f"💡 Embedding {len(texts)} chunks for {doc_meta['doc_id']}")

        embeddings = embed_documents(texts)

        upsert_chunks(index, chunks, embeddings, base_meta=doc_meta)

    print(f"✅ Ingest complete. Docs processed: {[d['doc_id'] for d in docs_to_process]}")
