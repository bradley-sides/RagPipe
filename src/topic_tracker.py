from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def track_topic_evolution(topic, structured_transcripts):
    examples = "\n\n".join(f"### {name} ###\n{output}" for name, output in structured_transcripts)

    prompt = f"""
You are an earnings analyst reviewing quarterly earnings transcripts.

Below are excerpts from company transcripts that mention the topic: **{topic}**.
For each quarter, summarize how the company discusses the topic in 2–4 sentences.
Focus on what's mentioned, how it's framed, and how it has evolved.
Include the name of each quarter clearly and keep responses concise and readable. 
You may cite the most important references.

--- Transcript Mentions ---
{examples}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content