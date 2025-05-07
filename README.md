# final_project/README.py

# 🎬 YouTube QA Bot

A smart Streamlit application that lets you interact with any YouTube video using natural language.  
The system downloads the video, extracts the audio, transcribes it using Whisper or Faster-Whisper, stores the text in a vector database (Chroma), and allows GPT-powered Q&A.

---

## 🚀 Features

- 🎧 **Audio Download** from YouTube via `yt-dlp`
- 🧠 **Transcription** using Whisper / Faster-Whisper with automatic model selection
- ✂️ **Text Splitting** with `RecursiveCharacterTextSplitter`
- 🔍 **Embeddings** using OpenAI's `text-embedding-ada-002`
- 🗃️ **Vector Database** with `Chroma`
- 📄 **Summarization** using GPT models
- 💬 **Q&A Agent** built on LangChain and OpenAI
- 🌐 **Streamlit Web Interface** with styled chat

---

## 📦 Installation

