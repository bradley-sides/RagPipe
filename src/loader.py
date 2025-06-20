from langchain_community.document_loaders import PyPDFLoader
import os
def load_pdf(path):
    full = os.path.join(os.getcwd(), path)
    return PyPDFLoader(full).load()