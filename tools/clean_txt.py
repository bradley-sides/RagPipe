#!/usr/bin/env python3

import os
import re

# Where your raw .txt files live
TXT_DIR = "./text/"
CLEAN_DIR = "./text_clean/"

os.makedirs(CLEAN_DIR, exist_ok=True)

def clean_text(raw_text):
    # Remove multiple newlines
    text = re.sub(r"\n+", "\n", raw_text)

    # Join single newlines that should be spaces
    lines = text.split("\n")
    new_lines = []
    buffer = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if re.search(r"[.?!,:]$", stripped):
            buffer += " " + stripped
            new_lines.append(buffer.strip())
            buffer = ""
        else:
            buffer += " " + stripped

    if buffer.strip():
        new_lines.append(buffer.strip())

    return "\n\n".join(new_lines)

# Loop through all .txt files
for file_name in os.listdir(TXT_DIR):
    if file_name.endswith(".txt"):
        txt_path = os.path.join(TXT_DIR, file_name)
        clean_path = os.path.join(CLEAN_DIR, file_name)

        with open(txt_path, "r", encoding="utf-8") as f:
            raw = f.read()

        cleaned = clean_text(raw)

        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"✅ Cleaned: {clean_path}")

print("🏁 All done! Check your /text_clean folder.")