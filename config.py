# final_project/config.py

from langchain.agents import AgentType
from langchain.callbacks.manager import CallbackManager

# 🔥 Default value if automatic selection is not used
WHISPER_MODEL_SIZE = "small"

# 🔥 Rules for model selection based on audio duration (in minutes)
WHISPER_MODEL_RULES = {
    "short": {"max": 10, "model": "small"},
    "medium": {"max": 90, "model": "base"},
    "long": {"max": float("inf"), "model": "medium"},
}

def select_whisper_model(duration_minutes: float) -> str:
    if duration_minutes <= WHISPER_MODEL_RULES["short"]["max"]:
        return WHISPER_MODEL_RULES["short"]["model"]
    elif duration_minutes <= WHISPER_MODEL_RULES["medium"]["max"]:
        return WHISPER_MODEL_RULES["medium"]["model"]
    else:
        return WHISPER_MODEL_RULES["long"]["model"]

# 🔁 Automatically map Whisper model to the appropriate GPT model
WHISPER_TO_GPT_MAP = {
    "base": "gpt-3.5-turbo",
    "small": "gpt-4",
    "medium": "gpt-4"
}

def select_gpt_model_by_whisper(whisper_model: str) -> str:
    return WHISPER_TO_GPT_MAP.get(whisper_model, OPENAI_MODEL_NAME)

# 🔥 GPT settings
OPENAI_MODEL_NAME = "gpt-3.5-turbo"

# 🔥 Embedding model settings
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"

# 🔥 Agent type setting
AGENT_TYPE = AgentType.CONVERSATIONAL_REACT_DESCRIPTION

# 🔥 General settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
