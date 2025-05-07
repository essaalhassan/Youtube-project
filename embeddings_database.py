# final_project/embeddings_database.py

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings          # ← Correct import after package split
from langchain_community.vectorstores import Chroma    # ← Same library reused in app.py

from final_project.config import EMBEDDING_MODEL_NAME

# 🔥 Load environment variables
load_dotenv()


def create_vectorstore(docs, persist_dir: str):
    """
    Builds a Chroma VectorStore from a list of text chunks, saves it in persist_dir,
    and returns the object (can be queried directly or closed and used via the path).
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    store.persist()                      # Writes index + metadata files
    return store
