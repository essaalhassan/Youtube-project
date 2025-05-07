import os
from pydub.utils import mediainfo
from pydub import AudioSegment

def get_audio_duration(filepath: str) -> float:
    """
    Reads the duration of an audio file in minutes.
    """
    info = mediainfo(filepath)
    duration_sec = float(info['duration'])
    return duration_sec / 60  # Convert from seconds to minutes


def split_audio(filepath: str, chunk_duration_min: int = 20, output_dir: str = None) -> list:
    """
    Splits a long audio file into chunks of a specified duration (in minutes),
    and saves the chunks into a specified folder (e.g., cache/audio/).

    Returns:
        A list of full paths for all the split audio parts.
    """
    audio = AudioSegment.from_file(filepath)
    duration_ms = len(audio)
    chunk_duration_ms = chunk_duration_min * 60 * 1000  # Duration per chunk in milliseconds

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(filepath), "split_parts")
    os.makedirs(output_dir, exist_ok=True)

    chunks = []
    for i in range(0, duration_ms, chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        chunk_filename = f"{os.path.splitext(os.path.basename(filepath))[0]}_part{i // chunk_duration_ms}.wav"
        chunk_path = os.path.join(output_dir, chunk_filename)
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks
