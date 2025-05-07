from huggingface_hub import snapshot_download
import os

os.makedirs("models", exist_ok=True)

# Download faster-whisper-base
snapshot_download(
    repo_id="guillaumekln/faster-whisper-base",
    local_dir="models/faster-whisper-base",
    resume_download=True
)

# Download faster-whisper-small
snapshot_download(
    repo_id="guillaumekln/faster-whisper-small",
    local_dir="models/faster-whisper-small",
    resume_download=True
)

# Download faster-whisper-medium
snapshot_download(
    repo_id="guillaumekln/faster-whisper-medium",
    local_dir="models/faster-whisper-medium",
    resume_download=True
)

print("✅ All models downloaded successfully.")
