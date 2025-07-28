from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def track_guidance_completion(structured_transcripts):
    # Format: List[Tuple[filename, structured_output_text]]
    examples = "\n\n".join(f"### {name} ###\n{output}" for name, output in structured_transcripts)

    prompt = f"""
            You are an earnings analyst. Each section below is a structured transcript from a quarterly earnings call.
            Your task is to analyze how the company's forward-looking guidance played out in the following quarters.

            Create a clear line-item timeline showing:
            - What guidance was issued
            - Whether the company reiterated, missed, or exceeded it in later quarters
            - Any changes in tone or expectations

            End with a brief summary about guidance follow-through quality.

            {examples}
            """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content