import json
import os

BASE_DIR = os.path.dirname(__file__)  # This is /RagPipe/src
DOCS_PATH = os.path.join(BASE_DIR, "docs.json")

with open(DOCS_PATH) as f:
    DOCS = json.load(f)