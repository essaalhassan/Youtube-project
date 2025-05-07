# deployment/app.py

import os
import logging
from textwrap import shorten

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from yt_dlp import YoutubeDL

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from openai import OpenAI
from langsmith import Client

from final_project.download_audio import download_audio_from_youtube
from final_project.transcribe_audio import transcribe_audio
from final_project.text_processing import split_text
from final_project.embeddings_database import create_vectorstore
from final_project.summarization import generate_summary
from final_project.cache_utils import save_cache, load_cache, generate_cache_key
from final_project.agent import build_agent
from final_project.config import select_gpt_model_by_whisper

# ─── Load API keys and initialize clients
dotenv_path = find_dotenv(".env", raise_error_if_not_found=True)
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

missing = [k for k, v in {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "LANGCHAIN_API_KEY": LANGCHAIN_API_KEY,
    "LANGCHAIN_TRACING_V2": LANGCHAIN_TRACING_V2,
    "LANGCHAIN_PROJECT": LANGCHAIN_PROJECT,
}.items() if not v]
if missing:
    raise EnvironmentError(f"Missing keys in .env: {', '.join(missing)}")

os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

openai_client = OpenAI(api_key=OPENAI_API_KEY)
langsmith_client = Client()

# ─── Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Streamlit app configuration
st.set_page_config(page_title="YouTube QA Bot", page_icon="🎥")

# ─── Custom CSS for chat bubbles
st.markdown("""
<style>
.user-bubble{background:#ffcccc;color:#000;padding:8px 12px;border-radius:12px;
             max-width:85%;display:inline-block;margin-bottom:4px;}
.bot-bubble{background:#000;color:#fff;padding:8px 12px;border-radius:12px;
            max-width:85%;display:inline-block;margin-bottom:4px;}
</style>
""", unsafe_allow_html=True)

# ─── Ensure all required cache directories exist
for d in ("audio", "transcripts", "summaries", "vectorstore", "temp"):
    os.makedirs(f"cache/{d}", exist_ok=True)

# ─── Default values for session state
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("brief", "No summary yet.")

def get_video_metadata(url: str) -> dict:
    try:
        with YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        logger.error(f"Failed to fetch metadata: {e}")
        return {"title": "—", "duration": "—", "short_desc": "—"}
    return {
        "title": info.get("title", "—"),
        "duration": (
            f"{int(info['duration']) // 60}:{int(info['duration']) % 60:02d}"
            if info.get("duration") else "—"
        ),
        "short_desc": shorten(info.get("description", ""), 150, placeholder="...")
    }

# ─── User input section
st.title("🎬 YouTube AI Assistant")
with st.form("url_form"):
    youtube_url = st.text_input("🔗 YouTube URL:", placeholder="Enter URL here")
    gpt_model_choice = st.selectbox(
        "🤖 Select GPT Model (optional)",
        ["", "gpt-3.5-turbo", "gpt-4"],
        index=0,
        help="Leave it empty to automatically choose based on Whisper model"
    )
    process = st.form_submit_button("ENTER")

# ─── Main video processing workflow
if process and youtube_url:
    with st.spinner("Processing…"):
        try:
            key = generate_cache_key(youtube_url)
            cached = load_cache(key)

            if cached:
                transcript, summary, vs_dir = cached
                vectorstore = Chroma(
                    persist_directory=vs_dir,
                    embedding_function=OpenAIEmbeddings()
                )
                chosen_model = gpt_model_choice or "gpt-4"
            else:
                wav = download_audio_from_youtube(youtube_url)
                wav_p = f"cache/audio/{key}.wav"
                os.replace(wav, wav_p)

                progress_bar = st.progress(0)
                transcript, whisper_used = transcribe_audio(wav_p, progress_bar)
                progress_bar.progress(1.0)

                chosen_model = gpt_model_choice or select_gpt_model_by_whisper(whisper_used)

                with open(f"cache/transcripts/{key}.txt", "w", encoding="utf-8") as f:
                    f.write(transcript)

                try:
                    summary = generate_summary(transcript, model_name=chosen_model)
                except Exception as e:
                    msg = str(e).lower()
                    if "insufficient_quota" in msg or "429" in msg:
                        st.error("OpenAI quota exceeded. Summary generation is not available at the moment.")
                        summary = "—"
                    else:
                        raise

                with open(f"cache/summaries/{key}.txt", "w", encoding="utf-8") as f:
                    f.write(summary)

                vs_dir = f"cache/vectorstore/{key}"
                vectorstore = create_vectorstore(split_text(transcript), persist_dir=vs_dir)
                save_cache(key, transcript, summary, vs_dir)

            agent = build_agent(vectorstore, model_name=chosen_model)
            st.session_state.qa_bot = agent
            st.session_state.metadata = get_video_metadata(youtube_url)
            st.session_state.brief = summary
            st.session_state.chat_history = []

        except Exception as e:
            logger.error(f"Processing error: {e}")
            st.error(f"Error during processing: {e}")

# ─── Display video title and summary
if "metadata" in st.session_state:
    md = st.session_state.metadata
    st.markdown(f"**Title:** {md['Title']}  \n**Duration:** {md['duration']}")
    st.markdown("**Brief:**")
    st.write(st.session_state.brief)

# ─── Display past user-bot interactions
if st.session_state.chat_history:
    st.markdown("---")
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(f"<div class='user-bubble'>{q}</div>", unsafe_allow_html=True)
        with st.chat_message("assistant"):
            st.markdown(f"<div class='bot-bubble'>{a}</div>", unsafe_allow_html=True)

# ─── Handle user input for QA
if "qa_bot" in st.session_state:
    user_question = st.chat_input("❓ Ask your question about the video content:")
    if user_question:
        with st.chat_message("user"):
            st.markdown(f"<div class='user-bubble'>{user_question}</div>", unsafe_allow_html=True)

        if "title" in user_question.lower():
            answer_text = st.session_state.metadata.get("title", "Sorry, the title is not available.")
        elif any(k in user_question.lower() for k in ("summary", "brief")):
            answer_text = st.session_state.brief
        else:
            with st.spinner("✍️ Generating Answer..."):
                try:
                    answer_text = st.session_state.qa_bot.run(user_question)
                except Exception as e:
                    msg = str(e).lower()
                    if "insufficient_quota" in msg or "429" in msg:
                        st.error("OpenAI quota exceeded. Answer generation is not available at the moment.")
                        answer_text = "Sorry, I can't generate an answer right now."
                    else:
                        raise

        with st.chat_message("assistant"):
            st.markdown(f"<div class='bot-bubble'>{answer_text}</div>", unsafe_allow_html=True)

        st.session_state.chat_history.append((user_question, answer_text))
        with open("qa_log.txt", "a", encoding="utf-8") as f:
            f.write(f"Q: {user_question}\nA: {answer_text}\n{'-'*40}\n")
