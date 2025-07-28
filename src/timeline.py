from src.vectorstore import init_index
from src.query import run_query
from src.rag import build_timeline_prompt, generate_answer

def run_timeline_query(query: str, company: str = None) -> str:
    index = init_index()
    chunks = run_query(index, query, top_k=40, company=company)

    if not chunks:
        return "❌ No relevant transcript data found."

    prompt = build_timeline_prompt(chunks, query)
    return generate_answer(prompt)