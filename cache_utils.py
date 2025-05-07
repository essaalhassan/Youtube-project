# final_project/cache_utils.py
"""
Cache for each video's data:
- Stores the transcript
- The summary
- The Chroma vectorstore path (vectorstore_dir)

❱ The actual vectorstore is saved to disk inside create_vectorstore,
  so we don't store the object inside Pickle.
"""

import os
import hashlib
import pickle
from typing import Tuple, Optional

# Single directory for all small cache files (.pkl)
_META_DIR = "cache/meta"
os.makedirs(_META_DIR, exist_ok=True)


# ─────────────────────────── General Functions ───────────────────────────
def generate_cache_key(url: str) -> str:
    """
    Converts a YouTube URL into a consistent 16-character MD5 key.
    Example:  8ffdefbdec95 = generate_cache_key("https://youtu.be/...")
    """
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:16]


def _meta_path(key: str) -> str:
    """Returns the full path to the Pickle file for the given key."""
    return os.path.join(_META_DIR, f"{key}.pkl")


def save_cache(key: str, transcript: str, summary: str, vectorstore_dir: str) -> None:
    """
    Writes a Pickle file containing only paths and text data.
    """
    payload = {
        "transcript": transcript,
        "summary": summary,
        "vectorstore_dir": vectorstore_dir,
    }
    with open(_meta_path(key), "wb") as fp:
        pickle.dump(payload, fp)


def load_cache(key: str) -> Optional[Tuple[str, str, str]]:
    """
    If the cache file exists, returns (transcript, summary, vectorstore_dir),
    otherwise returns None.
    """
    path = _meta_path(key)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fp:
        data = pickle.load(fp)
    return data["transcript"], data["summary"], data["vectorstore_dir"]
