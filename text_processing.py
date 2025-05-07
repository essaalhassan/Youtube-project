# final_project/text_processing.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from final_project.config import CHUNK_SIZE, CHUNK_OVERLAP

def split_text(text: str):
    """Split a large text into smaller chunks."""
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,           # Each chunk is 1000 characters
        chunk_overlap=CHUNK_OVERLAP      # Slight overlap to preserve meaning
    )
    
    docs = splitter.create_documents([text])
    return docs
