# my_podcast_proj/common_image.py

import modal

shared_volume = modal.Volume.from_name("podcast_gen_volume", create_if_missing=True)
print(f"Using volume: podcast_gen_volume (created if it was missing)")

common_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")  
    .pip_install(
        "torch==2.5.1",  # THIS HAS BEEN SOLVED NOW BUT ORIGINALLY - temp fix to avoid stricter security check by torch: https://github.com/suno-ai/bark/pull/619
        "PyPDF2",
        "python-fasthtml==0.12.0",
        "langchain",
        "langchain-community",
        "openai-whisper",
        "beautifulsoup4",
        "requests",
        "pydub",
        "nltk",
        "tqdm",
        "scipy",
        "transformers==4.46.1",
        "accelerate==1.2.1",
        "sumy>=0.11.0",  # Add sumy for dedicated summarization
        "git+https://github.com/suno-ai/bark.git"
    )
)
