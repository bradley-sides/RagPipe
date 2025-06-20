from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def build_prompt(chunks_with_meta, query):

    """
    chunks_with_meta: list of strings, each already prefixed
                      with metadata info, e.g.
                      "[NVDA Q1 FY2026 • 2025-09-15]\n<text>"
    """

    context = "\n\n".join(chunks_with_meta)
    if not context:
        raise ValueError("No context provided for the query.")
    
    return f"""You are an expert analyst of quarterly financial reports. Based only on the following earnings call 
    transcript and provided context, answer the question below. Do NOT guess or fabricate any numbers 
    not explicitly shown in the context. If exact values are not available, say so. 
    Be sure that the numbers you provide are associated with the correct fiscal quarter and year. Be sure 
    that the numbers you provide are associated with the corret statistic (e.g. revenue, EPS, etc.).
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

Context:
{context}

Question:
{query}

Answer:"""

def generate_answer(prompt):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content