import modal
from input_gen import app as input_app
from scripts_gen import app as script_app
from audio_gen import app as audio_app

app = modal.App("gen-podcast")

# Ensure script and audio apps share the same volume
shared_volume = modal.Volume.lookup("podcast_gen_volume")

# Also make sure the Bark models volume is available
try:
    bark_volume = modal.Volume.lookup("bark_models")
    print("Found existing bark_models volume")
except modal.exception.NotFoundError:
    print("Warning: bark_models volume not found. Audio generation may fail.")
    print("Please run download_bark_models.py first")

app.include(input_app)
app.include(script_app)
app.include(audio_app)
