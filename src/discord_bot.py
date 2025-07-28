import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
import asyncio
import aiohttp
import json
import re

# pipeline imports
from src.vectorstore import init_index
from src.query import run_query
from src.oneshot_parser import oneshot_structure_transcript
from src.vectorstore import init_index
from src.ingest import ingest_documents
from src.timeline import run_timeline_query
from src.pdf_export import render_text_to_pdf
from src.guidance_tracker import track_guidance_completion
from src.topic_tracker import track_topic_evolution
from src.topic_tracker import track_topic_evolution
from src.pdf_export import render_text_to_pdf
from src.rag import build_prompt, generate_answer, answer_query_from_chunks
BASE_DIR = os.path.dirname(__file__)   # /src
DOCS_PATH = os.path.join(BASE_DIR, "docs.json")
# load env vars
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", "YOUR_CHANNEL_ID"))
DATA_DIR = "./text"

# pipeline wrapper
class RAGBot:
    def __init__(self):
        self.index = init_index()
        self.history = []

    def chat(self, user_query, company=None, top_k=10):
        answer = run_query(
            self.index,
            user_query,
            top_k=top_k,
            history=self.history,
            company=company
        )
        if answer:
            self.history.append((user_query, answer))
        return answer

# discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

rag_pipeline = RAGBot()

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

# helper to safely send long messages in chunks
async def send_long_message(ctx, content, max_len=2000):
    if isinstance(content, list):
        lines = content  # Treat each list item as its own block
    else:
        lines = content.split('\n')

    chunk = ""
    for line in lines:
        # If adding this line would exceed the limit, send what we have
        if len(chunk) + len(line) + 1 > max_len:
            await ctx.send(chunk)
            chunk = ""
        chunk += line + "\n"
    if chunk.strip():
        await ctx.send(chunk)

@bot.command(name="ingest")
async def ingest_docs(ctx, doc_id=None):
    if ctx.channel.id != CHANNEL_ID:
        await ctx.send("This bot can’t be used in this channel.")
        return

    index = init_index()

    # Remind user if they forget the argument
    if not doc_id:
        #await ctx.send("⚠️ Please specify a doc ID to ingest, or use `!ingest all` to re-ingest everything.")
        return

    #await ctx.send(f"⚙️ Ingesting `{doc_id}` — this may take a moment...")

    # Run the blocking sync code in a thread so the bot stays alive
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, ingest_documents, index, doc_id)

    #await ctx.send(f"✅ Ingest for `{doc_id}` complete! Ready for queries.")

@bot.command(name="upload")
async def upload_document(ctx):
    if ctx.channel.id != CHANNEL_ID:
        await ctx.send("This bot can’t be used in this channel.")
        return

    if not ctx.message.attachments:
        await ctx.send("⚠️ Please attach a file with your command.")
        return

    for attachment in ctx.message.attachments:
        file_name = attachment.filename
        save_path = os.path.join(DATA_DIR, file_name)

        #await ctx.send(f"📥 Downloading `{file_name}`...")
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url) as resp:
                if resp.status == 200:
                    with open(save_path, 'wb') as f:
                        f.write(await resp.read())
                    #await ctx.send(f"✅ Saved `{file_name}` to `{save_path}`.")
                else:
                    await ctx.send(f"❌ Failed to download `{file_name}`.")

    #await ctx.send("📌 New file(s) saved in `./data`. Run `!meta` to add them to the index and then !ingest to add them to the vector store")
    #await ctx.send("!meta formatting: COMPANY QUARTER FISCAL_YEAR CALL_DATE")
    #await ctx.send("example: !meta NVTS Q1 2024 2024-05-09")

@bot.command(name="meta")
async def add_metadata(ctx, company, quarter, fiscal_year, call_date):
    if ctx.channel.id != CHANNEL_ID:
        await ctx.send("This bot can’t be used in this channel.")
        return

    new_meta = {
        "file_path": f"text/{company.upper()}_{quarter.upper()}_{fiscal_year}.txt",
        "doc_id": f"{company.lower()}_{quarter.lower()}_fy{fiscal_year}",
        "company": company.upper(),
        "fiscal_year": int(fiscal_year),
        "quarter": quarter.upper(),
        "call_date": call_date
    }
    print(f" {os.path.abspath(DOCS_PATH)}")

    # ✅ Always read existing data
    if os.path.exists(DOCS_PATH):
        with open(DOCS_PATH, "r") as f:
            docs = json.load(f)
    else:
        docs = []

    docs.append(new_meta)

    # ✅ Write the updated list back
    with open(DOCS_PATH, "w") as f:
        json.dump(docs, f, indent=2)

    #await ctx.send(f"✅ Added metadata to `docs.json`. Ready for `!ingest`.")
    #await ctx.send(f"run !ingest company_quarter_fyYear for a single file or !ingest all for batch processing")
    #await ctx.send(f"ex. !ingest nvts_q2_fy2024")

@bot.command(name="outline")
async def oneshot(ctx, *, filename: str):
    path = os.path.join(DOCS_FOLDER, filename)
    if not os.path.isfile(path):
        await ctx.send(f"❌ File `{filename}` not found.")
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            transcript = f.read()

        result = oneshot_structure_transcript(transcript)

        # Generate a clean PDF filename
        base_name = filename.replace(".txt", "").strip()
        pdf_filename = f"{base_name}.pdf"

        # Convert result to PDF
        render_text_to_pdf(result, output_path=pdf_filename)

        await ctx.send(file=discord.File(pdf_filename))

    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")

@bot.command(name="timeline")
async def timeline(ctx, *, user_query):

    try:
        # Extract --company flag
        company_match = re.search(r"--company\\s+(\\w+)", user_query)
        company = company_match.group(1).upper() if company_match else None

        # Remove the flag from the query
        cleaned_query = re.sub(r"--company\\s+\\w+", "", user_query).strip()

        result = run_timeline_query(cleaned_query, company=company)

        if len(result) > 1900:
            with open("timeline_output.txt", "w", encoding="utf-8") as f:
                f.write(result)
            await ctx.send(" ", file=discord.File("timeline_output.txt"))
        else:
            await ctx.send(f"```\n{result}\n```")

    except Exception as e:
        await ctx.send(f"❌ Error generating timeline: {str(e)}")

# Path to your document folder
DOCS_FOLDER = "./text"

# Regex to match formats like OPEN_Q3_2024 or 2024_Q2_OPEN
doc_pattern = re.compile(r"(?:([A-Z]{1,5})_Q[1-4]_(\d{4})|(\d{4})_Q[1-4]_([A-Z]{1,5}))")

@bot.command(name="list")
async def list_docs(ctx, ticker: str):
    ticker = ticker.upper()
    matched_docs = []

    for filename in os.listdir(DOCS_FOLDER):
        if not filename.endswith(".txt"):
            continue

        match = doc_pattern.match(filename.replace(".txt", ""))
        if not match:
            continue

        # Extract tickers from both pattern types
        tickers = [group for group in [match.group(1), match.group(4)] if group]
        if ticker in tickers:
            matched_docs.append(filename)

    if matched_docs:
        await ctx.send(f" " + "\n".join(sorted(matched_docs)))
    else:
        await ctx.send(f"❌ No documents found for `{ticker}`.")

@bot.command(name="submit")
async def submit(ctx, company: str, quarter: str, fiscal_year: str, call_date: str):
    if ctx.channel.id != CHANNEL_ID:
        await ctx.send("This bot can’t be used in this channel.")
        return

    if not ctx.message.attachments:
        await ctx.send("⚠️ Please attach the `.txt` transcript.")
        return

    try:
        # Step 1: Save the file
        attachment = ctx.message.attachments[0]
        filename = f"{company.upper()}_{quarter.upper()}_{fiscal_year}.txt"
        save_path = os.path.join("text", filename)

        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url) as resp:
                if resp.status == 200:
                    with open(save_path, 'wb') as f:
                        f.write(await resp.read())
                else:
                    await ctx.send(f"❌ Failed to download `{filename}`.")
                    return

        # Step 2: Write metadata to docs.json
        new_meta = {
            "file_path": save_path,
            "doc_id": f"{company.lower()}_{quarter.lower()}_fy{fiscal_year}",
            "company": company.upper(),
            "fiscal_year": int(fiscal_year),
            "quarter": quarter.upper(),
            "call_date": call_date
        }

        if os.path.exists(DOCS_PATH):
            with open(DOCS_PATH, "r") as f:
                docs = json.load(f)
        else:
            docs = []

        docs.append(new_meta)

        with open(DOCS_PATH, "w") as f:
            json.dump(docs, f, indent=2)

        # Step 3: Ingest into vector store
        index = init_index()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, ingest_documents, index, new_meta["doc_id"])

        #await ctx.send(f"✅ Transcript for `{company.upper()} {quarter.upper()} {fiscal_year}` uploaded, tagged, and ingested.")

    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")

@bot.command(name="track_guidance")
async def track_guidance(ctx, ticker: str):
    ticker = ticker.upper()
    all_files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".txt") and ticker in f]
    
    if len(all_files) < 2:
        await ctx.send(f"❌ Not enough transcripts found for `{ticker}` — need at least 2.")
        return

    # Sort by quarter + year
    def sort_key(filename):
        parts = filename.replace(".txt", "").split("_")
        # Try parsing both common formats
        try:
            # Format: Q1_2024_NVTS.txt
            if parts[0].startswith("Q"):
                quarter = int(parts[0][1:])
                fy = int(parts[1])
            # Format: 2024_Q1_NVTS.txt
            elif parts[1].startswith("Q"):
                quarter = int(parts[1][1:])
                fy = int(parts[0])
            else:
                raise ValueError("Unknown format")
        except Exception:
            fy = 0
            quarter = 0
        return (fy, quarter)
    sorted_files = sorted(all_files, key=sort_key)[-4:]

    structured_transcripts = []
    for fname in sorted_files:
        with open(os.path.join(DOCS_FOLDER, fname), "r", encoding="utf-8") as f:
            raw = f.read()
        structured = oneshot_structure_transcript(raw)
        structured_transcripts.append((fname, structured))

    markdown_report = track_guidance_completion(structured_transcripts)
    pdf_path = render_text_to_pdf(markdown_report, output_path="guidance_report.pdf")

    await ctx.send(f"📊 Guidance tracking report for `{ticker}`", file=discord.File(pdf_path))

@bot.command(name="track_topic")
async def track_topic(ctx, *, arg_string: str):
    # Match --company TICKER followed by the topic (rest of the string)
    match = re.match(r"--company\s+([A-Za-z]{1,5})\s+(.+)", arg_string.strip())
    if not match:
        await ctx.send("❌ Format: `!track_topic --company <TICKER> <TOPIC>`")
        return

    ticker, topic = match.group(1).upper(), match.group(2).strip()
    #await ctx.send(f"📚 Analyzing `{ticker}` transcripts for commentary on: **{topic}**...")

    try:
        all_files = os.listdir(DOCS_FOLDER)
        company_files = [
            f for f in all_files
            if ticker in f.upper() and f.endswith(".txt")
        ]

        if not company_files:
            await ctx.send(f"❌ No transcripts found for `{ticker}` in `{DOCS_FOLDER}`.")
            return

        structured_transcripts = []
        for fname in sorted(company_files):
            full_path = os.path.join(DOCS_FOLDER, fname)
            with open(full_path, "r", encoding="utf-8") as f:
                raw_transcript = f.read()
            structured = oneshot_structure_transcript(raw_transcript)
            structured_transcripts.append((fname, structured))

        analysis = track_topic_evolution(topic, structured_transcripts)

        pdf_path = render_text_to_pdf(analysis, output_path="topic_evolution_report.pdf")
        await ctx.send(file=discord.File(pdf_path))

    except Exception as e:
        await ctx.send(f"❌ Error during topic tracking: {str(e)}")

@bot.command(name='find')
async def find_chunks(ctx):
    if ctx.channel.id != CHANNEL_ID:
        await ctx.send("This bot can’t be used in this channel.")
        return

    user_input = ctx.message.content.split()
    if len(user_input) < 2:
        await ctx.send("Please ask a full question, e.g., `!find --company NVTS How did they perform?`")
        return

    # Extract company flag if present
    company = None
    args = user_input[1:]  # skip the command prefix

    if '--company' in args:
        idx = args.index('--company')
        try:
            company = args[idx + 1]
            del args[idx:idx + 2]
        except IndexError:
            await ctx.send("⚠️ You must provide a company after --company.")
            return
    quarter = None
    fiscal_year = None

    if '--quarter' in args:
        idx = args.index('--quarter')
        try:
            q_str = args[idx + 1].upper()  # e.g. Q2
            fy_str = args[idx + 2]         # e.g. 2025

            if not q_str.startswith("Q") or not fy_str.isdigit():
                await ctx.send("⚠️ Format must be `--quarter Qx YYYY`, e.g. `--quarter Q2 2025`.")
                return

            fiscal_year = int(fy_str)
            quarter = q_str
            # Remove all 3 tokens: --quarter, Q2, 2025
            del args[idx:idx + 3]
        except IndexError:
            await ctx.send("⚠️ Format must be `--quarter Qx YYYY`, e.g. `--quarter Q2 2025`.")
            return
        
    question = ' '.join(args)
    print(quarter)
    loop = asyncio.get_running_loop()
    chunks = await loop.run_in_executor(
        None,
        lambda: run_query(rag_pipeline.index, question, top_k=10, company=company, quarter=quarter, fiscal_year=fiscal_year)
    )

    if not chunks:
        await ctx.send("Sorry, I couldn't find anything relevant.")
        return

    await send_long_message(ctx, "\n\n".join(chunks))

@bot.command(name='ask')
async def ask_question(ctx):
    if ctx.channel.id != CHANNEL_ID:
        await ctx.send("This bot can’t be used in this channel.")
        return

    user_input = ctx.message.content.split()
    if len(user_input) < 2:
        await ctx.send("Please ask a full question, e.g., `!ask --company NVTS How did they perform?`")
        return

    # Parse args
    company = None
    args = user_input[1:]

    if '--company' in args:
        idx = args.index('--company')
        try:
            company = args[idx + 1]
            del args[idx:idx + 2]
        except IndexError:
            await ctx.send("⚠️ You must provide a company after --company.")
            return
    quarter = None
    fiscal_year = None

    if '--quarter' in args:
        idx = args.index('--quarter')
        try:
            q_str = args[idx + 1].upper()  # e.g. Q2
            fy_str = args[idx + 2]         # e.g. 2025

            if not q_str.startswith("Q") or not fy_str.isdigit():
                await ctx.send("⚠️ Format must be `--quarter Qx YYYY`, e.g. `--quarter Q2 2025`.")
                return

            fiscal_year = int(fy_str)
            quarter = q_str
            # Remove all 3 tokens: --quarter, Q2, 2025
            del args[idx:idx + 3]
        except IndexError:
            await ctx.send("⚠️ Format must be `--quarter Qx YYYY`, e.g. `--quarter Q2 2025`.")
            return
    question = ' '.join(args)
    loop = asyncio.get_running_loop()

    # Step 1: get top chunks
    chunks = await loop.run_in_executor(None, lambda: run_query(rag_pipeline.index, question, top_k=10, company=company, quarter=quarter, fiscal_year=fiscal_year))

    if not chunks:
        await ctx.send("Sorry, I couldn't find any relevant context.")
        return

    # Step 2: build final answer from chunks
    answer = await loop.run_in_executor(None, lambda: answer_query_from_chunks(chunks, question))

    await send_long_message(ctx, answer)

bot.run(DISCORD_TOKEN)