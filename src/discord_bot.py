import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
import asyncio

# pipeline imports
from src.vectorstore import init_index
from src.query import run_query

# load env vars
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", "YOUR_CHANNEL_ID"))

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
# ask command
@bot.command(name='ask')
async def ask_question(ctx):
    if ctx.channel.id != CHANNEL_ID:
        await ctx.send("This bot can’t be used in this channel.")
        return

    user_input = ctx.message.content.split()
    if len(user_input) < 2:
        await ctx.send("Please ask a full question, e.g., `!ask --company NVTS How did they perform?`")
        return

    # Extract company flag if present
    company = None
    args = user_input[1:]  # skip the command prefix

    if '--company' in args:
        idx = args.index('--company')
        try:
            company = args[idx + 1]
            # Remove the flag and value from args
            del args[idx:idx + 2]
        except IndexError:
            await ctx.send("⚠️ You must provide a company after --company.")
            return

    question = ' '.join(args)

    await ctx.send(f"Query: `{question}`")
    if company:
        await ctx.send(f"Using company filter: `{company}`")

    loop = asyncio.get_running_loop()
    # Pass the company to your pipeline!
    answer = await loop.run_in_executor(None, rag_pipeline.chat, question, company)

    if not answer:
        await ctx.send("Sorry, I couldn't find an answer.")
        return

    await send_long_message(ctx, answer)

from src.vectorstore import init_index
from src.ingest import ingest_documents
import aiohttp

DATA_DIR = "./data"

@bot.command(name="ingest")
async def ingest_docs(ctx, doc_id=None):
    if ctx.channel.id != CHANNEL_ID:
        await ctx.send("This bot can’t be used in this channel.")
        return

    index = init_index()

    # Remind user if they forget the argument
    if not doc_id:
        await ctx.send("⚠️ Please specify a doc ID to ingest, or use `!ingest all` to re-ingest everything.")
        return

    await ctx.send(f"⚙️ Ingesting `{doc_id}` — this may take a moment...")

    # Run the blocking sync code in a thread so the bot stays alive
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, ingest_documents, index, doc_id)

    await ctx.send(f"✅ Ingest for `{doc_id}` complete! Ready for queries.")

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

        await ctx.send(f"📥 Downloading `{file_name}`...")
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url) as resp:
                if resp.status == 200:
                    with open(save_path, 'wb') as f:
                        f.write(await resp.read())
                    await ctx.send(f"✅ Saved `{file_name}` to `{save_path}`.")
                else:
                    await ctx.send(f"❌ Failed to download `{file_name}`.")

    await ctx.send("📌 New file(s) saved in `./data`. Run `!meta` to add them to the index and then !ingest to add them to the vector store")
    await ctx.send("!meta formatting: COMPANY QUARTER FISCAL_YEAR CALL_DATE")
    await ctx.send("example: !meta NVTS Q1 2024 2024-05-09")

import json
BASE_DIR = os.path.dirname(__file__)   # /src
DOCS_PATH = os.path.join(BASE_DIR, "docs.json")

@bot.command(name="meta")
async def add_metadata(ctx, company, quarter, fiscal_year, call_date):
    if ctx.channel.id != CHANNEL_ID:
        await ctx.send("This bot can’t be used in this channel.")
        return

    new_meta = {
        "file_path": f"data/{company.upper()}_{quarter.upper()}_{fiscal_year}.pdf",
        "doc_id": f"{company.lower()}_{quarter.lower()}_fy{fiscal_year}",
        "company": company.upper(),
        "fiscal_year": int(fiscal_year),
        "quarter": quarter.upper(),
        "call_date": call_date
    }
    print(f"📄 Writing docs.json to: {os.path.abspath(DOCS_PATH)}")

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

    await ctx.send(f"✅ Added metadata to `docs.json`. Ready for `!ingest`.")
    await ctx.send(f"run !ingest company_quarter_fyYear for a single file or !ingest all for batch processing")
    await ctx.send(f"ex. !ingest nvts_q2_fy2024")
bot.run(DISCORD_TOKEN)