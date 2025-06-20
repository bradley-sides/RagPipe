# src/utils.py
import re
from langchain.schema import Document
def clean_pages(pages):
    """
    Collapse stray line breaks within paragraphs so each Document.page_content
    is a normal paragraph, not one-word lines.
    """
    cleaned = []
    for doc in pages:
        txt = doc.page_content
        # 1) Replace single newlines (not double) with space
        txt = re.sub(r'(?<!\n)\n(?!\n)', ' ', txt)
        # 2) Collapse any remaining multiple whitespace/newlines
        txt = re.sub(r'\s+', ' ', txt).strip()
        cleaned.append(Document(page_content=txt, metadata=doc.metadata))
    return cleaned