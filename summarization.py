# final_project/summarization.py

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from final_project.config import OPENAI_MODEL_NAME
import streamlit as st

def generate_summary(text: str, model_name: str = None) -> str:
    """
    Generates a summary of the given text using a GPT model.
    You can manually pass a model_name or use the default from config.
    """
    model_to_use = model_name or OPENAI_MODEL_NAME
    summarizer = ChatOpenAI(model_name=model_to_use)

    prompt = PromptTemplate.from_template(
        "Summarize the following text in a short paragraph:\n\n{text}"
    )
    summary_chain = prompt | summarizer

    try:
        response = summary_chain.invoke({"text": text})
        if hasattr(response, "content"):
            return response.content
        elif isinstance(response, dict) and "content" in response:
            return response["content"]
        else:
            return str(response)

    except Exception as e:
        # If the error has http_status = 429, show a friendly message
        if getattr(e, "http_status", None) == 429:
            st.error("Sorry, you have exceeded the OpenAI usage limit. Summary generation is temporarily unavailable.")
            return "—"
        raise
