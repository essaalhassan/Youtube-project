# final_project/agent.py

from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from final_project.config import OPENAI_MODEL_NAME

def build_agent(vectorstore, model_name: str = None):
    """
    Builds a RetrievalQA chain based on the vector store,
    so that it answers only based on the video's transcript.
    """
    llm = ChatOpenAI(model_name=model_name or OPENAI_MODEL_NAME)
    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",        # or "map_reduce" or "refine" based on your preference
        retriever=retriever,
        return_source_documents=False
    )

    return qa_chain
