import json

# Path to your JSON
DOCS_PATH = "./src/docs.json"

with open(DOCS_PATH, "r") as f:
    docs = json.load(f)

# Replace .pdf with .txt in file_path
for doc in docs:
    if doc["file_path"].endswith(".pdf"):
        doc["file_path"] = doc["file_path"].replace("data/", "text/").replace(".pdf", ".txt")

# Overwrite the same file or save to a new one if you want backup
with open(DOCS_PATH, "w") as f:
    json.dump(docs, f, indent=2)

print("✅ All file_path entries updated from PDF to TXT!")