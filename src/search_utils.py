import os
import heapq
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

'''
    search_utils contains two functions:

        1. optimize_query: Passes user query in and returns query optimized for vector search

        2. rerank_chunks: Takes metadata and text associated with each matched vector store result
                          and ranks based on qualities specified to trim the less relevant results 
                          from the knowledge base passed to the main model.
'''

# Query Rewrite for optimized vector searhc
def optimize_query(user_query: str) -> str:
    """
    Turn a loose user question into a concise retrieval query.
    """
    prompt = f"""
    You are an expert search engineer. Rewrite the user's question into a concise, keyword-focused 
    retrieval query. Keep it very short. It is important to include any time frame mentioned by the user. 
    For example, if the user wants results spanning FY 2024 Q1 - 2025 Q2, include all quarter names and 
    years in the query individually (FY 2025 Q1, FY 2025 Q2, FY 2025 Q3, FY 2025 Q4, FY 2026 Q1)

    User question:
    \"{user_query}\"

    Retrieval query:"""
    
    response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature =0
            )
    # Model returns rewritten query
    return response.choices[0].message.content.strip()

def rerank_chunks(chunks: list[dict], user_query: str, top_k: int = 5) -> list[dict]:
    """
    Given a list of Pinecone match dicts (each with 'metadata' and 'text'),
    ask the LLM to score each on relevance to user_query, then
    return the top_k by that score.

    It is paramount that your top ranking criteria is accuracy. If I ask for 2026 related data, you
    must rank the 2026 data higher than 2025 data, even if the 2025 data is more relevant to the question.
    If I ask for revenue, you must rank chunks with revenue data higher than those without. 
    If I ask for revenue in 2026, you must rank chunks with 2026 revenue data higher than those with 2025 revenue data.
    
    It is also critical that you are aware of when in TIME the Fiscal Year refers to. For example, if I want Q1 FY 2026 information on 
    supply chain forecast, I am interested in the date of the call onward, which is May 2025. Be very careful about this as it important.
    """
    scored = []
    for chunk in chunks:
        text = chunk["metadata"]["text"]
        prompt = f"""
        On a scale from 0 (irrelevant) to 1 (perfectly relevant),
        how relevant is this excerpt to answering: "{user_query}"?

        Excerpt:
        {text}

        Score:"""

        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature = 0
        )
        try:
            score = float(resp.choices[0].message.content.strip())
        except ValueError:
            score = 0.0
        scored.append((score, chunk))

    # Pick the top_k by the LLM’s score
    top = heapq.nlargest(top_k, scored, key = lambda x: x[0])
    return [chunk for score, chunk in top]