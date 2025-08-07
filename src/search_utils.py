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
def optimize_query(user_query: str, memory = "") -> str:
    """
    Turn a loose user question into a concise retrieval query.
    """
    prompt = f"""
    You are an expert search engineer. Rewrite the user's question into a concise, keyword-focused 
    retrieval query. Keep it very short. It is important to include any time frame mentioned by the user. 
    For example, if the user wants results spanning FY 2024 Q1 - 2025 Q2, include all quarter names and 
    years in the query individually (FY 2025 Q1, FY 2025 Q2, FY 2025 Q3, FY 2025 Q4, FY 2026 Q1)

    Here is the conversation so far, if it is empty, disregard, but if there is a summarized conversation, adapt
    your query to address the user's new question in the context of the previous conversation:
     
    {memory}

    User question:
    \"{user_query}\"

    Retrieval query:"""
    
    response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature =0
            )
    # Model returns rewritten query
    return response.choices[0].message.content.strip()

def rerank_chunks(chunks: list[dict], user_query: str, top_k: int = 10) -> list[dict]:
    prompt = f"""
    Given a list of Pinecone match dicts (each with 'metadata' and 'text'),
    ask the LLM to score each on relevance to user_query, then
    return the top_k by that score.

    The most IMPORTANT THING is to rank the correct TIME PERIOD the highest. If I ask for 2024 Q1, you cannot rank 2024 Q2 higher than ANY Q1 2024 excerpt. Any instance will be penalized heavily.

    It is paramount that your top ranking criteria is accuracy. If I ask for 2026 related data, you
    must rank the 2026 data higher than 2025 data, even if the 2025 data is more relevant to the question.
    If I ask for revenue, you must rank chunks with revenue data higher than those without. 
    If I ask for revenue in 2026, you must rank chunks with 2026 revenue data higher than those with 2025 revenue data.
    
    It is also paramount that you are aware of when in TIME the Fiscal Year refers to. For example, if I want Q1 FY 2026 information on 
    supply chain forecast, I am interested in the date of the call onward, which is May 2025. Be very careful about this as it important.
    
    If I ask for information from a time range, you MUST provide one reference from EACH quarter in that range before making more additions.
    These must all be included in the top {top_k} results after you rank.

    These is is the query: {user_query}

    These are the chunks: {''.join(f"### Chunk {i+1}:\n{chunk['metadata']['text']}\n" for i, chunk in enumerate(chunks))}

    Return ONLY the top {top_k} chunk numbers as a raw JSON array of integers, e.g., [3, 1, 5, 2]. Do not include any explanation or extra text.
    """
    resp = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        import json
        top_indices = json.loads(resp.choices[0].message.content)
        return [chunks[i - 1] for i in top_indices if 0 < i <= len(chunks)]
    except Exception as e:
        print(f"⚠️ Failed to parse rerank response: {e}")
        return chunks[:top_k]
