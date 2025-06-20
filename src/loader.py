from langchain_community.document_loaders import PyPDFLoader
import os
# Uses langchain's pdf loader to get the pdf as a Document object.
def load_pdf(path):
    full = os.path.join(os.getcwd(), path)
    return PyPDFLoader(full).load()