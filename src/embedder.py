import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

embedding_model = OpenAIEmbeddings()

def embed_documents(texts):
    return embedding_model.embed_documents(texts)

def embed_query(query):
    return embedding_model.embed_query(query)