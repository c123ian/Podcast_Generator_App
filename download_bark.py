import modal
import os
import shutil
from pathlib import Path

app = modal.App("bark_model_downloader")

MODELS_DIR = "/bark_models"  # Volume mount path for Bark models

# Create or look up the volume for Bark models
try:
    volume = modal.Volume.from_name("bark_models", create_if_missing=True)
except modal.exception.NotFoundError:
    volume = modal.Volume.persisted("bark_models")

# Create an image with necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")  # Add git first
    .pip_install(
        "torch==2.5.1",  # Match version in common_image.py
        "numpy",
        "scipy",
        "tqdm",
        "git+https://github.com/suno-ai/bark.git"
    )
)

MINUTES = 60
HOURS = 60 * MINUTES

app = modal.App(image=image)

@app.function(volumes={MODELS_DIR: volume}, timeout=2 * HOURS)
def download_bark_models(force_download=False):
    """Download Bark models to the persistent volume"""
    import os
    import shutil
    
    # Import Bark here after installation
    from bark import preload_models
    
    # Reload volume to ensure we see up-to-date content
    volume.reload()
    
    # Default Bark models are stored in this location
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
    default_bark_dir = os.path.join(xdg_cache_home, "suno", "bark_v0")
    
    # Create volume directory structure
    volume_bark_dir = os.path.join(MODELS_DIR, "bark_v0")
    os.makedirs(volume_bark_dir, exist_ok=True)
    
    # Check if we already have models in our volume
    existing_files = []
    if os.path.exists(volume_bark_dir):
        existing_files = os.listdir(volume_bark_dir)
    
    # Check if any of the important model files are present
    important_model_files = ["text_2.pt", "coarse_2.pt", "fine_2.pt"]
    model_files_exist = any(file in existing_files for file in important_model_files)
    
    if force_download or not model_files_exist:
        print(f"Downloading Bark models...")
        
        # Set model path to default location for download
        os.environ["XDG_CACHE_HOME"] = xdg_cache_home
        
        # Download the models
        preload_models()
        
        # List all files in the default cache location
        if os.path.exists(default_bark_dir):
            model_files = os.listdir(default_bark_dir)
            print(f"Models downloaded. Found {len(model_files)} files.")
            
            # Copy all model files to the volume
            for model_file in model_files:
                source_path = os.path.join(default_bark_dir, model_file)
                target_path = os.path.join(volume_bark_dir, model_file)
                
                # Skip hidden files and directories (like .cache)
                if model_file.startswith('.'):
                    print(f"Skipping hidden file/directory: {model_file}")
                    continue
                
                # Check if it's a file or directory and handle accordingly
                if os.path.isfile(source_path):
                    print(f"Copying file: {model_file}...")
                    shutil.copy2(source_path, target_path)
                elif os.path.isdir(source_path):
                    print(f"Skipping directory: {model_file}")
                    # Optionally copy directories using copytree if needed:
                    # if os.path.exists(target_path):
                    #     shutil.rmtree(target_path)
                    # shutil.copytree(source_path, target_path)
                else:
                    print(f"Unknown file type: {model_file}")
            
            print(f"All model files copied to volume at {volume_bark_dir}")
            
            # Directly copy the specific model files we know we need
            for model_file in important_model_files:
                source_path = os.path.join(default_bark_dir, model_file)
                target_path = os.path.join(volume_bark_dir, model_file)
                if os.path.exists(source_path) and os.path.isfile(source_path):
                    print(f"Ensuring critical model file is copied: {model_file}")
                    shutil.copy2(source_path, target_path)
                else:
                    print(f"Warning: Critical model file not found: {model_file}")
        else:
            print(f"Error: Default model directory {default_bark_dir} not found after download.")
    else:
        print(f"Bark models already exist in volume. Found {len(existing_files)} files.")
    
    # List files in the volume for verification
    if os.path.exists(volume_bark_dir):
        files = os.listdir(volume_bark_dir)
        print(f"Files in volume: {files}")
    
    # Commit changes to the volume
    volume.commit()
    print("Volume committed with Bark models")

@app.local_entrypoint()
def main(force_download: bool = False):
    """Entry point for downloading Bark models"""
    print("Starting Bark model download...")
    download_bark_models.remote(force_download)
    print("Bark model download complete")
