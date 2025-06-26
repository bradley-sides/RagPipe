import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

#
'''
    embedder: contains two functions to embed both docs (on insertion to database)
              and query (on search)
'''
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

embedding_model = OpenAIEmbeddings()

def embed_documents(texts, batch_size=16):
    print(f"Embedding {len(texts)} documents in batches of {batch_size}")

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            embeddings = embedding_model.embed_documents(batch)
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"[Batch {i}-{i+batch_size}] Error embedding batch: {e}")
            raise
    return all_embeddings

def embed_query(query):
    return embedding_model.embed_query(query)