#!/usr/bin/env python3

import os
import json
from PyPDF2 import PdfReader

# === CONFIG ===
# Uses your docs.json as source of truth
DOCS_PATH = "./src/docs.json"

# Optional: your PDFs can be in data/ or anywhere else
PDF_BASE_DIR = "./"
TXT_DIR = "./text/"

os.makedirs(TXT_DIR, exist_ok=True)

# Load your metadata
with open(DOCS_PATH) as f:
    docs = json.load(f)

print(f"✅ Loaded {len(docs)} docs from {DOCS_PATH}")

for doc in docs:
    pdf_path = os.path.join(PDF_BASE_DIR, doc["file_path"])
    txt_filename = os.path.basename(pdf_path).replace(".pdf", ".txt")
    txt_path = os.path.join(TXT_DIR, txt_filename)

    print(f"📄 Processing: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"⚠️ PDF file not found: {pdf_path} — Skipping.")
        continue

    reader = PdfReader(pdf_path)
    full_text = ""

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += text + "\n"
        else:
            print(f"⚠️ Page {i+1} might be empty or scanned.")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"✅ Saved: {txt_path}")

print("🏁 All done! Check your /text folder.")