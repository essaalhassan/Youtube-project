# final_project/transcribe_audio.py

from faster_whisper import WhisperModel
from final_project.config import select_whisper_model
from final_project.audio_utils import get_audio_duration, split_audio
import os

def transcribe_audio(file_path: str, progress_bar=None) -> tuple:
    """
    Transcribes an audio file to text using Faster-Whisper,
    with automatic model selection, chunking of long files, and optional Streamlit progress bar.
    Returns: (full transcript, Whisper model name used)
    """
    # Calculate audio duration (in minutes)
    duration = get_audio_duration(file_path)

    # Select Whisper model based on duration
    model_size = select_whisper_model(duration)

    # Load model on CPU to avoid CUDA/cuDNN errors
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # If the audio is longer than 30 minutes → split into 10-minute chunks
    if duration > 30:
        cache_audio_dir = os.path.join("cache", "audio", "chunks")
        os.makedirs(cache_audio_dir, exist_ok=True)

        audio_parts = split_audio(
            file_path,
            chunk_duration_min=10,
            output_dir=cache_audio_dir
        )

        all_text = []
        total_parts = len(audio_parts)

        for idx, part in enumerate(audio_parts, start=1):
            # Transcribe each part
            segments, _ = model.transcribe(
                part,
                beam_size=1,
                vad_filter=True
            )
            text = " ".join(seg.text for seg in segments)
            all_text.append(text)

            # Update progress bar if provided
            if progress_bar:
                progress_bar.progress(idx / total_parts)

        # Combine all chunks into full transcript
        return "\n".join(all_text), model_size

    else:
        # Direct transcription for short files
        segments, _ = model.transcribe(
            file_path,
            beam_size=1,
            vad_filter=True
        )
        transcript = " ".join(seg.text for seg in segments)

        # Ensure progress bar reaches 100%
        if progress_bar:
            progress_bar.progress(1.0)

        return transcript, model_size
