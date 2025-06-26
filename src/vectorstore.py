from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
from uuid import uuid4

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME = os.getenv("INDEX_NAME", "earnings-rag")
EMBEDDING_DIM = 1536

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set.")
client = Pinecone(api_key=PINECONE_API_KEY)

def init_index():
    if not any(i.name == INDEX_NAME for i in client.list_indexes()):
        client.create_index(
            name = INDEX_NAME,
            dimension =EMBEDDING_DIM,
            metric = "cosine",
            spec = ServerlessSpec(cloud = "aws", region =PINECONE_ENV)
        ) 
    return client.Index(INDEX_NAME)

def upsert_chunks(index,chunks, embeddings, base_meta: dict):
    payloads = []
    # Upsert list of chunks to pinecone database
    # chunk.metadata has things like page number, source, etc.
    # base_meta should contain document_id, quarter, year, company name, etc.

    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks and embeddings must match.")
    for i, chunk in enumerate(chunks):
        payloads.append({
            "id": f"{base_meta['doc_id']}-{i}",
            "values": embeddings[i],
            "metadata": {
                **base_meta,
                **chunk.metadata,
                "text": chunk.page_content,
            }
        })
    index.upsert(payloads)

def query_index(index, query_vector, top_k = 5, metadata_filter = None):
    if metadata_filter:
        print(f"Applying metadata filter: {metadata_filter}")  # 🔍
    return index.query(vector= query_vector, 
                       top_k = top_k, 
                       include_metadata =True, 
                       filter= metadata_filter
                    )