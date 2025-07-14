from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document  # Make sure you have this import

import os
# Uses langchain's pdf loader to get the pdf as a Document object.
def load_pdf(path):
    full = os.path.join(os.getcwd(), path)
    return PyPDFLoader(full).load()

# 🟢 Load TXT (returns a Document list too)
def load_txt(path):
    full = os.path.join(os.getcwd(), path)
    with open(full, "r", encoding="utf-8") as f:
        text = f.read()
    return [Document(page_content=text)]