from openai import OpenAI
from dotenv import load_dotenv
import os
from collections import defaultdict
import re
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def build_prompt(chunks_with_meta, query):
    context = "\\n\\n".join(chunks_with_meta)
    if not context:
        raise ValueError("No context provided for the query.")

    return f"""You are an expert analyst of quarterly financial reports. Based only on the following earnings call 
    transcript and provided context, answer the question below. Do NOT guess or fabricate any numbers 
    not explicitly shown in the context. If exact values are not available, say so. 
    Be sure that the numbers you provide are associated with the correct fiscal quarter and year. Be sure 
    that the numbers you provide are associated with the correct statistic (e.g. revenue, EPS, etc.).
    Incorrect or fabricated numbers will result in a penalty.

    Information passed in this form: [NVDA | QQ2 FY2025 | 2024-08-28 | page 1/16] references the company,
    fiscal quarter and year, call date, and page number in the earnings call transcript. This is important
    data for you to know in order to answer the question correctly. When constructing your answer, be mindful
    that the data you use is temporally accurate and relevant to the question asked. If asked about a trend or
    evolution of a statistic, consider the temporal context of the data provided.

    It is also important that you understand what time period is referenced by the Quarter and Fiscal Year based on the
    date of the call and the quarter it references. For instance, if I'm asking about Q1 2026 FY guidance, I want to pull
    from DOCUMENTS from Q1 2026, but information INSIDE could reference May 2025 and beyond, because that is the time period.

    If asked a question about a specific period of time, prioritize the text tagged with that period.

    YOU MUST FORMAT YOUR ANSWER ONLY WITH PLAIN TEXT, YOU CAN USE BOLD AND REGULAR TEXT AND BULLTS BUT NO TABLES.
Context:
{context}

Question:
{query}

Answer:"""

def generate_answer(prompt):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def summarize_memory(history, max_turns=10):
    memory = ""
    for q, a in history[-max_turns:]:
        memory += f"User: {q}\\nAssistant: {a}\\n"
    return memory


def build_timeline_prompt(chunks_with_meta, query):
    context = "\n\n".join(chunks_with_meta)

    return f"""You are a financial analyst tasked with constructing a timeline of how a company addressed a specific issue over time.

            Below is a question and 40 excerpts from earnings calls. Each excerpt starts with a label like:
            [OPEN | Q2 FY2023 • 2023-05-10]

            These contain metadata including the fiscal quarter, year, and date. Use these to group insights by quarter and arrange them chronologically.

            Question:
            {query}

            Excerpts:
            {context}

            Instructions:
            - Group the excerpts by fiscal quarter in order
            - Under each quarter, write 2–4 bullet points based on relevant excerpts
            - If a quarter lacks relevant data, state so
            - Conclude with a brief summary of trends or evolution across the timeline
            """


def answer_query_from_chunks(chunks_with_meta: list[str], user_query: str) -> str:
    """
    Uses retrieved chunks and the user query to construct and return a final answer.
    """
    if not chunks_with_meta:
        return "Sorry, I couldn't find relevant context to answer that."

    try:
        prompt = build_prompt(chunks_with_meta, user_query)
        print("=== Final Prompt to OpenAI ===")
        print(prompt[:1000])  # avoid printing full prompt if too long
        return generate_answer(prompt)
    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        return "An error occurred while generating the answer."