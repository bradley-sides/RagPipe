from langchain.text_splitter import RecursiveCharacterTextSplitter

"""
    chunker contains one function:
        
        1. chunk_documents: takes in the document and splits it into chunks of size 500 with
                            with 100 overlapping characters to hold the context of info.
"""

def chunk_documents(documents, chunk_size = 500, chunk_overlap = 100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap =chunk_overlap,
        #separators =["\n\n", "\n", " ", ""] #tiered
        separators=[""] #dumbed it down for txt processing
        )
    return splitter.split_documents(documents)