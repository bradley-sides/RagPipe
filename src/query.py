from src.embedder import embed_query
from src.vectorstore import init_index, upsert_chunks, query_index, client, INDEX_NAME
from src.rag import build_prompt, generate_answer, summarize_memory
from src.search_utils import optimize_query, rerank_chunks

def run_query(index, user_query, top_k = 10, history = None, company = None, quarter = None, fiscal_year = None):
    history = history or []

    memory = summarize_memory(history)
    memory_section = f"""The following is a summary of the prior conversation. Use it to maintain context and continuity in your answer if appropriate.

    {memory}

    """
    optimized_q = optimize_query(user_query, memory = memory)
    print(f"Optimized query: {optimized_q}\\n")
    search_filter = {"company": company.upper()} if company else None
    if quarter:
        search_filter["quarter"] = quarter.upper()
    if fiscal_year:
        search_filter["fiscal_year"] = int(fiscal_year)
    results = query_index(
        index,
        embed_query(optimized_q),
        top_k=top_k * 4,
        metadata_filter = search_filter
    )
    matches = results.get("matches", [])
    if not matches:
        print("No matches found for the query.")
        return None

    print(f"Retrieved {len(matches)} candidates, now reranking...\\n")
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
        print("\\n")
    print("__________________________________________________")

    chunks_with_meta = []
    for match in matches:
        md = match["metadata"]
        label = f"[{md['company']} | Q{md['quarter']} FY{int(md['fiscal_year'])} • {md['call_date']}]"
        text = md.get("text", "")
        chunks_with_meta.append(f"{label}\\n{text}")

    print("Generating answer from top texts...")

    #memory = summarize_memory(history)
    #memory_section = f"""The following is a summary of the prior conversation. Use it to maintain context and continuity in your answer if appropriate.

    #{memory}

    #"""
    #full_prompt = memory_section + build_prompt(chunks_with_meta, user_query)
    #print(full_prompt)

    #answer = generate_answer(full_prompt)
    print("DEBUGGING")
    print()
    print(type(chunks_with_meta))
    print(chunks_with_meta[:1] if chunks_with_meta else "None")
    return chunks_with_meta