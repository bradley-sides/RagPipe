from src.loader import load_pdf
from src.chunker import chunk_documents
from src.embedder import embed_documents, embed_query
from src.vectorstore import init_index, upsert_chunks, query_index, client, INDEX_NAME
from src.config import DOCS
from src.utils import clean_pages
from src.query import run_query

'''
    ingest documents and send them to Pinecone vector store
'''
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

