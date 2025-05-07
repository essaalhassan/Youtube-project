# final_project/download_audio.py

import yt_dlp
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

def download_audio_from_youtube(url: str) -> str:
    """Download audio as a temporary WAV file from a YouTube URL."""
    temp_dir = tempfile.mkdtemp()
    output_template = os.path.join(temp_dir, 'audio.%(ext)s')

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '8',  # 8,16,32
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        logger.error(f"Error downloading audio from {url}: {e}")
        raise

    wav_file_path = output_template.replace('%(ext)s', 'wav')
    return wav_file_path
