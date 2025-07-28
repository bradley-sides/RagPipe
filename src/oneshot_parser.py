from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def oneshot_structure_transcript(new_transcript: str) -> str:
    with open("src/one-shot-text/transcript.txt", "r", encoding="utf-8") as f:
        example_transcript = f.read()

    with open("src/one-shot-text/notes.txt", "r", encoding="utf-8") as f:
        example_output = f.read()

    prompt = f"""
    You are a financial assistant trained to organize earnings transcripts into structured sections.

    Here is an example of how a transcript should be structured:

    ### EXAMPLE TRANSCRIPT ###
    {example_transcript}

    ### EXAMPLE STRUCTURED OUTPUT ###
    {example_output}

    Now apply the same structure to this new transcript:

    ### NEW TRANSCRIPT ###
    {new_transcript}

    ### NEW STRUCTURED OUTPUT ###
    """
    # Send to OpenAI
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a precise financial analyst trained to organize transcripts into structured notes."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
