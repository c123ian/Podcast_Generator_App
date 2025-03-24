import modal
import os
import sqlite3
import uuid
import time
import random
import json
from typing import Optional
import torch
import base64
import asyncio
from fasthtml.common import *
import PyPDF2

import whisper

from langchain_community.document_loaders import WebBaseLoader
from common_image import common_image, shared_volume

from starlette.responses import FileResponse, StreamingResponse, HTMLResponse, RedirectResponse

import modal

# Ensure these functions reference the correct deployed app
generate_script = modal.Function.from_name("gen-podcast", "generate_script")
generate_audio = modal.Function.from_name("gen-podcast", "generate_audio")

# Global config at module level - single consistent limit
MAX_CONTENT_CHARS = 45000  # ~18,750 tokens

class BaseIngestor:
    """Base class for all ingestors"""
    def validate(self, source: str) -> bool:
        pass

    def extract_text(self, source: str, max_chars: int = MAX_CONTENT_CHARS) -> Optional[str]:
        pass
    
    def truncate_with_warning(self, text: str, max_chars: int) -> str:
        """Intelligently truncate text with warning message"""
        if len(text) <= max_chars:
            return text
            
        truncated = text[:max_chars]
        print(f"⚠️ Content truncated from {len(text)} to {max_chars} characters")
        
        # Add an explanatory note at the end
        truncation_note = "\n\n[Note: The original content was truncated due to length limitations.]"
        truncated = truncated[:max_chars - len(truncation_note)] + truncation_note
        
        return truncated


class PDFIngestor(BaseIngestor):
    """PDF ingestion - core functionality"""
    def validate(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            print(f"Error: File not found at path: {file_path}")
            return False
        if not file_path.lower().endswith('.pdf'):
            print("Error: File is not a PDF")
            return False
        return True

    def extract_text(self, file_path: str, max_chars: int = MAX_CONTENT_CHARS) -> Optional[str]:
        if not self.validate(file_path):
            return None
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                print(f"Processing PDF with {num_pages} pages...")
                extracted_text = []
                total_chars = 0
                for page_num in range(num_pages):
                    page_text = pdf_reader.pages[page_num].extract_text()
                    if page_text:
                        extracted_text.append(page_text)
                        total_chars += len(page_text)
                        print(f"Processed page {page_num + 1}/{num_pages}, total chars: {total_chars}")
                        if total_chars > max_chars * 1.1:  # Read slightly more than needed
                            print(f"Reached character limit at page {page_num + 1}/{num_pages}")
                            break
                full_text = "\n".join(extracted_text)
                return self.truncate_with_warning(full_text, max_chars)
        except PyPDF2.PdfReadError:
            print("Error: Invalid or corrupted PDF file")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return None


class WebsiteIngestor(BaseIngestor):
    """Website ingestion using LangChain's WebBaseLoader"""
    def validate(self, url: str) -> bool:
        if not url.startswith(('http://', 'https://')):
            print("Error: Invalid URL format")
            return False
        return True

    def extract_text(self, url: str, max_chars: int = MAX_CONTENT_CHARS) -> Optional[str]:
        if not self.validate(url):
            return None
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            extracted_text = "\n".join([doc.page_content for doc in documents])
            print(f"Extracted {len(extracted_text)} chars from website: {url}")
            return self.truncate_with_warning(extracted_text, max_chars)
        except Exception as e:
            print(f"An error occurred while extracting from website: {str(e)}")
            return None


class AudioIngestor(BaseIngestor):
    """Audio ingestion using OpenAI's Whisper model"""
    def __init__(self, model_type: str = "base"):
        self.model_type = model_type
        self.model = whisper.load_model(self.model_type)

    def validate(self, audio_file: str) -> bool:
        if not os.path.exists(audio_file):
            print(f"Error: Audio file not found at path: {audio_file}")
            return False
        if not audio_file.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
            print("Error: Unsupported audio format. Supported formats are .mp3, .wav, .flac, .m4a")
            return False
        return True

    def extract_text(self, audio_file: str, max_chars: int = MAX_CONTENT_CHARS) -> Optional[str]:
        if not self.validate(audio_file):
            return None
        try:
            result = self.model.transcribe(audio_file)
            transcription = result["text"]
            print(f"Transcribed {len(transcription)} chars from audio file: {audio_file}")
            return self.truncate_with_warning(transcription, max_chars)
        except Exception as e:
            print(f"An error occurred during audio transcription: {str(e)}")
            return None


class TextIngestor(BaseIngestor):
    """Simple ingestor for .txt or .md files."""
    def validate(self, file_path: str) -> bool:
        if not os.path.isfile(file_path):
            print(f"Error: File not found at path: {file_path}")
            return False
        if not (file_path.lower().endswith('.txt') or file_path.lower().endswith('.md')):
            print("Error: File is not .txt or .md")
            return False
        return True

    def extract_text(self, file_path: str, max_chars: int = MAX_CONTENT_CHARS) -> Optional[str]:
        if not self.validate(file_path):
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"Extracted {len(text)} chars from text file: {file_path}")
            return self.truncate_with_warning(text, max_chars)
        except Exception as e:
            print(f"Error reading text file: {e}")
            return None


class IngestorFactory:
    """Factory to create appropriate ingestor based on input type"""
    @staticmethod
    def get_ingestor(input_type: str, **kwargs) -> Optional[BaseIngestor]:
        input_type = input_type.lower()
        if input_type == "pdf":
            return PDFIngestor()
        elif input_type == "website":
            return WebsiteIngestor()
        elif input_type == "audio":
            return AudioIngestor(**kwargs)
        elif input_type == "text":
            return TextIngestor()
        else:
            print(f"Unsupported input type: {input_type}")
            return None


# Define directories for persistent storage across container restarts
UPLOAD_DIR = "/data/uploads"       # Stores temporary uploaded files (PDFs, audio files, etc.)
OUTPUT_DIR = "/data/processed"     # Stores processed text content extracted from inputs
DB_PATH = "/data/injections.db"    # SQLite database file for tracking all podcast generation jobs
AUDIO_DIR = "/data/podcast_audio"  # Storage location for final generated podcast audio files
STATUS_DIR = "/data/status"        # Backup storage for job status in JSON format (fault tolerance)

# Ensure Directories Exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(STATUS_DIR, exist_ok=True)  # Create status directory

# Function to get standardized audio file path
def get_audio_file_path(injection_id):
    """Returns a standardized path for audio files based on injection_id"""
    return os.path.join(AUDIO_DIR, f"podcast_{injection_id}.wav")

# Function to save status to file as a fallback for database issues
def save_status_file(injection_id, status, notes=None, file_path=None):
    """Save status to a file as a fallback when database is unavailable"""
    if not injection_id:
        return
        
    status_file = os.path.join(STATUS_DIR, f"{injection_id}.json")
    status_data = {
        "id": injection_id,
        "status": status,
        "notes": notes,
        "file_path": file_path,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        with open(status_file, "w") as f:
            json.dump(status_data, f)
        print(f"✅ Saved status file for ID: {injection_id}")
    except Exception as e:
        print(f"⚠️ Error saving status file: {e}")

# Function to update injection status with retry logic
def update_injection_status(injection_id, status, notes=None, max_retries=5):
    """Update the status and optional notes for an injection with retry logic"""
    if not injection_id:
        return
    
    # Also save to file-based status system as backup
    audio_path = get_audio_file_path(injection_id)
    save_status_file(injection_id, status, notes, audio_path)
    
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=10.0)  # Longer timeout
            cursor = conn.cursor()
            
            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            
            # Update status and/or notes
            if notes:
                cursor.execute(
                    "UPDATE injections SET status = ?, processing_notes = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (status, notes, injection_id)
                )
            else:
                cursor.execute(
                    "UPDATE injections SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (status, injection_id)
                )
                
            conn.commit()
            conn.close()
            print(f"✅ Updated status to '{status}' for ID: {injection_id}")
            
            if notes:
                print(f"📝 Notes: {notes}")
                
            return
            
        except sqlite3.OperationalError as e:
            # Handle database lock errors
            if "database is locked" in str(e) and attempt < max_retries - 1:
                wait_time = 0.1 * (2 ** attempt) + random.random() * 0.1  # Exponential backoff
                print(f"⚠️ Database locked, retrying in {wait_time:.2f} seconds (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"⚠️ Error updating injection status after {attempt+1} attempts: {e}")
                break
        except Exception as e:
            print(f"⚠️ Error updating injection status: {e}")
            break
    
    # If we get here, all retries failed but we still have the file-based status

# Improved function to find audio files with multiple fallbacks
def find_audio_file(injection_id):
    """More robust audio file detection with multiple fallbacks"""
    # Try standard path first
    standard_path = get_audio_file_path(injection_id)
    if os.path.exists(standard_path):
        print(f"Found audio at standard path: {standard_path}")
        return standard_path
        
    # Try checking database
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5.0)
        cursor = conn.cursor()
        
        # Enable WAL mode
        cursor.execute("PRAGMA journal_mode=WAL;")
        
        cursor.execute(
            "SELECT processed_path FROM injections WHERE id = ?", 
            (injection_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] and os.path.exists(result[0]):
            print(f"Found audio from database path: {result[0]}")
            return result[0]
            
    except Exception as e:
        print(f"Error querying database for audio path: {e}")
            
    # Fallback: Search for any file containing the ID
    try:
        if os.path.exists(AUDIO_DIR):
            all_files = os.listdir(AUDIO_DIR)
            matching_files = [f for f in all_files if injection_id in f]
            
            if matching_files:
                found_path = os.path.join(AUDIO_DIR, matching_files[0])
                print(f"Found audio by directory search: {found_path}")
                return found_path
    except Exception as e:
        print(f"Error searching directory for audio: {e}")
            
    # Check status files as last resort
    try:
        status_file = os.path.join(STATUS_DIR, f"{injection_id}.json")
        if os.path.exists(status_file):
            with open(status_file, "r") as f:
                status_data = json.load(f)
                if "file_path" in status_data and os.path.exists(status_data["file_path"]):
                    print(f"Found audio from status file: {status_data['file_path']}")
                    return status_data["file_path"]
    except Exception as e:
        print(f"Error checking status file for audio: {e}")
            
    print(f"No audio file found for ID: {injection_id}")
    return None

# Create Modal App
app = modal.App("content_injection")

# Main FastHTML Server with defined routes
@app.function(
    image=common_image,
    volumes={"/data": shared_volume},
    cpu=2.0,
    timeout=3600
)
@modal.asgi_app()
def serve():
    """Main FastHTML Server"""
    # Set up the FastHTML app with required headers
    fasthtml_app, rt = fast_app(
        hdrs=(
            Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@3.9.2/dist/full.css"),
            Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"),
            Script(src="https://unpkg.com/htmx.org@1.9.10"),
        )
    )
    
    # Setup Database with improved concurrency settings
    def setup_database(db_path: str):
        """Initialize SQLite database with better concurrency settings"""
        # Check if directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Connect to database (will create if not exists)
        conn = sqlite3.connect(db_path, timeout=30.0)  # Increased timeout
        cursor = conn.cursor()
        
        # Enable WAL mode for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS injections (
                id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                input_type TEXT NOT NULL,
                processed_path TEXT,
                status TEXT DEFAULT 'pending',
                content_length INTEGER,
                processing_notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        return conn

    # IMPORTANT: Initialize database when module loads
    # Add this line after the directory creation code
    setup_database(DB_PATH)

    # Determine Input Type
    def get_input_type(filename: str) -> str:
        """Classifies input type based on filename or URL"""
        lower_filename = filename.lower()
        if lower_filename.endswith('.pdf'):
            return 'pdf'
        elif lower_filename.endswith(('.mp3', '.wav', '.m4a', '.flac')):
            return 'audio'
        elif lower_filename.startswith(('http://', 'https://')):
            return 'website'
        elif lower_filename.endswith(('.txt', '.md')):
            return 'text'
        else:
            raise ValueError(f"Unsupported file type: {filename}")

    # Process Uploaded Content
    def process_content(source_path: str, input_type: str, max_chars: int = 45000) -> Optional[str]:
        """Processes content using the appropriate ingestor"""
        ingestor = IngestorFactory.get_ingestor(input_type)
        if not ingestor:
            print(f"❌ No ingestor found for type: {input_type}")
            return None
        return ingestor.extract_text(source_path, max_chars)
    

    @rt("/inject", methods=["POST"])
    async def inject_content(request):
        """Handles content ingestion and starts the podcast generation pipeline."""
        form = await request.form()
        injection_id = uuid.uuid4().hex

        try:
            # Handle File Upload or URL
            if "content" in form and form["content"].filename:
                file_field = form["content"]
                original_filename = file_field.filename
                input_type = get_input_type(original_filename)
                save_path = os.path.join(UPLOAD_DIR, f"upload_{injection_id}{os.path.splitext(original_filename)[1]}")
                content = await file_field.read()
                with open(save_path, "wb") as f:
                    f.write(content)
            elif "url" in form and form["url"].strip():
                url = form["url"].strip()
                input_type = get_input_type(url)
                save_path = url
                original_filename = url
            else:
                return Div(
                    Div(
                        P("⚠️ Please select a file or provide a URL."),
                        cls="alert alert-warning"
                    ),
                    id="injection-status"
                )

            # Extract Text Content
            processed_text = process_content(save_path, input_type)
            if not processed_text:
                return Div(
                    Div(
                        P("❌ Failed to process content"),
                        cls="alert alert-error"
                    ),
                    id="injection-status"
                )

            # Set the standardized audio file path
            audio_file_path = get_audio_file_path(injection_id)

            # Insert record into database with content size info and initial status
            # Use retry logic to handle potential database locks
            for attempt in range(5):
                try:
                    conn = sqlite3.connect(DB_PATH, timeout=10.0)
                    cursor = conn.cursor()
                    
                    # Enable WAL mode for better concurrency
                    cursor.execute("PRAGMA journal_mode=WAL;")
                    cursor.execute("PRAGMA synchronous=NORMAL;")
                    
                    cursor.execute(
                        """INSERT INTO injections 
                        (id, original_filename, input_type, status, content_length, processing_notes, processed_path) 
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (injection_id, original_filename, input_type, "pending", 
                        len(processed_text), f"Content ingested: {len(processed_text)} chars", audio_file_path)
                    )
                    conn.commit()
                    conn.close()
                    break
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) and attempt < 4:
                        wait_time = 0.1 * (2 ** attempt) + random.random() * 0.1
                        print(f"⚠️ Database locked during insert, retrying in {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"⚠️ Error inserting record: {e}")
                        raise
                except Exception as e:
                    print(f"⚠️ Error inserting record: {e}")
                    raise

            # Also save to file-based status system
            save_status_file(injection_id, "pending", f"Content ingested: {len(processed_text)} chars", audio_file_path)

            # Update status to show script generation starting
            update_injection_status(injection_id, "processing", "Starting script generation...")

            # Proper function sequence with correctly coupled inputs/outputs
            print("🚀 Kicking off script generation...")
            
            # First run script generation and get the result
            script_data = generate_script.remote(processed_text)
            
            # Update status to show audio generation starting
            update_injection_status(injection_id, "processing", "Script generated. Starting audio generation...")
            
            # Then pass that directly to audio generation
            print("🔊 Kicking off audio generation...")
            generate_audio.spawn(script_data, injection_id)
            
            # Redirect to the podcast status page
            return RedirectResponse(f"/podcast-status/{injection_id}", status_code=303)

        except Exception as e:
            return Div(
                Div(
                    P(f"⚠️ Error processing content: {str(e)}"),
                    cls="alert alert-error"
                ),
                id="injection-status"
            )

    #################################################
    # Homepage Route - Upload Form
    #################################################
    @rt("/")
    def homepage():
        """Render upload form with status checker"""
        # DaisyUI styled file input
        upload_input = Input(
            type="file",
            name="content",
            accept=".pdf,.txt,.md,.mp3,.wav,.m4a,.flac",
            required=False,
            cls="file-input file-input-secondary w-full"
        )
        
        # DaisyUI styled URL input with prefix label
        url_input_container = Div(
            Span(cls="bg-base-300 px-3 py-2 rounded-l-lg"),
            Input(
                type="text",
                name="url",
                placeholder="https://",
                cls="grow px-3 py-2 bg-base-300 rounded-r-lg focus:outline-none"
            ),
            cls="flex items-center w-full"
        )
        
        # Side-by-side layout with divider
        side_by_side = Div(
            # Left card - File upload
            Div(
                Div(
                    upload_input,
                    cls="grid place-items-center p-4"
                ),
                cls="card bg-base-300 rounded-box grow"
            ),
            # Divider
            Div("OR", cls="divider divider-horizontal"),
            # Right card - URL input
            Div(
                Div(
                    url_input_container,
                    cls="grid place-items-center p-4"
                ),
                cls="card bg-base-300 rounded-box grow"
            ),
            cls="flex w-full"
        )
        
        # Content limits information
        content_info = Div(
            P(f"Maximum content length: {45000//1000}K characters (longer content will be truncated)",
              cls="text-sm text-center opacity-70 mt-2")
        )
        
        # Process button (no spinner initially)
        process_button = Button(
            "Process Content",
            cls="btn btn-primary w-full mt-4",
            type="submit",
            id="process-button"
        )

        # Add script to handle loading state
        loading_script = Script("""
        document.addEventListener('htmx:beforeRequest', function(evt) {
            if (evt.target.matches('form')) {
                // Find the submit button
                var btn = evt.target.querySelector('button[type="submit"]');
                if (btn) {
                    // Save the original text
                    btn.dataset.originalText = btn.textContent;
                    // Replace with loading spinner
                    btn.innerHTML = '<span class="loading loading-spinner loading-lg text-secondary"></span>';
                    btn.disabled = true;
                }
            }
        });
        
        document.addEventListener('htmx:afterRequest', function(evt) {
            if (evt.target.matches('form')) {
                // Find the submit button
                var btn = evt.target.querySelector('button[type="submit"]');
                if (btn && btn.dataset.originalText) {
                    // Restore original text
                    btn.innerHTML = btn.dataset.originalText;
                    btn.disabled = false;
                }
            }
        });
        
        // Fallback for non-HTMX form submissions
        document.addEventListener('DOMContentLoaded', function() {
            var uploadForm = document.getElementById('upload-form');
            if (uploadForm) {
                uploadForm.addEventListener('submit', function() {
                    var btn = document.getElementById('process-button');
                    if (btn) {
                        // Save the original text
                        btn.dataset.originalText = btn.textContent;
                        // Replace with loading spinner
                        btn.innerHTML = '<span class="loading loading-spinner loading-lg text-secondary"></span>';
                        btn.disabled = true;
                    }
                });
            }
        });
        """)

        upload_form = Form(
            side_by_side,
            content_info,
            process_button,
            loading_script,
            action="/inject",
            method="post",
            enctype="multipart/form-data",
            id="upload-form",
            hx_boost="true",  # Use HTMX to enhance the form
            hx_indicator="#process-button",  # Show loading state on this element
            cls="mb-6"
        )
        
        # Status checker form with DaisyUI styling
        status_form = Form(
            Div(
                Input(
                    type="text",
                    name="injection_id",
                    placeholder="Enter your podcast ID",
                    cls="input input-bordered w-full"
                ),
                Button("Check Status", cls="btn btn-secondary w-full mt-2"),
                cls="space-y-2"
            ),
            action="/status-redirect",
            method="get",
            cls="w-full"
        )

        return Title("AI Podcast Generator"), Main(
            Div(
                H1("Generate AI Podcast from Content", cls="text-2xl font-bold text-center mb-4"),
                P("Upload a file or provide a URL to process content for podcast generation.", 
                  cls="text-center mb-6"),
                upload_form,
                Div(id="injection-status", cls="my-4"),
                Div(
                    H1("Check Existing Podcast Status", cls="text-xl font-bold text-center mb-2"),
                    P("Already have a podcast ID? Check its status:", cls="text-center mb-4"),
                    status_form,
                    cls="mt-10 pt-8 border-t"
                ),
                cls="container mx-auto px-4 py-8 max-w-3xl"
            ),
            cls="min-h-screen bg-base-100"
        )
    
    

    #################################################
    # Status JSON API Endpoint for Lightweight Polling
    #################################################
    @rt("/podcast-status-api/{injection_id}")
    def podcast_status_api(injection_id: str):
        """API endpoint with improved reliability for checking podcast status"""
        # Always reload volume first
        for _ in range(3):  # Try up to 3 reloads
            shared_volume.reload()
            
            # First check if audio file exists directly (fastest path)
            audio_path = find_audio_file(injection_id)
            if audio_path:
                # Audio exists, return completed status
                return JSONResponse({
                    "status": "completed",
                    "notes": "Audio file found",
                    "is_completed": True,
                    "audio_exists": True,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            
            time.sleep(0.1)  # Brief pause between reloads
        
        # Then check file-based status (second fastest)
        status_file = os.path.join(STATUS_DIR, f"{injection_id}.json")
        if os.path.exists(status_file):
            try:
                with open(status_file, "r") as f:
                    status_data = json.load(f)
                    return JSONResponse({
                        "status": status_data.get("status", "processing"),
                        "notes": status_data.get("notes"),
                        "is_completed": status_data.get("status") == "completed",
                        "audio_exists": os.path.exists(status_data.get("file_path", "")),
                        "timestamp": status_data.get("updated_at")
                    })
            except Exception as e:
                print(f"Error reading status file: {e}")
    
        # Finally check database as last resort (with retries for locks)
        for attempt in range(3):
            try:
                conn = sqlite3.connect(DB_PATH, timeout=5.0)
                cursor = conn.cursor()
                
                # Enable WAL mode for better concurrency
                cursor.execute("PRAGMA journal_mode=WAL;")
                
                cursor.execute(
                    "SELECT status, processed_path, processing_notes, created_at FROM injections WHERE id = ?", 
                    (injection_id,)
                )
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    status, path, notes, created_at = result
                    is_completed = status == "completed"
                    audio_exists = False
                    
                    # Double check if audio exists
                    if is_completed:
                        audio_exists = path and os.path.exists(path)
                        
                        # If status is completed but can't find audio, do a final search
                        if not audio_exists:
                            audio_path = find_audio_file(injection_id)
                            audio_exists = audio_path is not None
                    
                    return JSONResponse({
                        "status": status,
                        "notes": notes,
                        "is_completed": is_completed,
                        "audio_exists": audio_exists,
                        "timestamp": created_at
                    })
                break
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < 2:
                    wait_time = 0.1 * (2 ** attempt) + random.random() * 0.1
                    print(f"⚠️ Database locked during status check, retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"⚠️ Error checking status in database: {e}")
            except Exception as e:
                print(f"⚠️ Error checking status in database: {e}")
                break
    
        return JSONResponse({
            "status": "unknown", 
            "message": "Podcast not found or database unavailable",
            "is_completed": False,
            "audio_exists": False
        })

    #################################################
    # Status Page with Hybrid Polling & Reload Logic
    #################################################
    @rt("/podcast-status/{injection_id}")
    def podcast_status(injection_id: str):
        """Status page that polls efficiently and reloads once when ready"""
        # First explicitly reload the volume multiple times to get latest changes
        for _ in range(3):
            shared_volume.reload()
            time.sleep(0.1)
        
        # Try the file-based status system first (faster)
        status_file = os.path.join(STATUS_DIR, f"{injection_id}.json")
        if os.path.exists(status_file):
            try:
                with open(status_file, "r") as f:
                    status_data = json.load(f)
                    status = status_data.get("status", "processing")
                    notes = status_data.get("notes", "Processing...")
                    created_at = status_data.get("updated_at", time.strftime("%Y-%m-%d %H:%M:%S"))
                    audio_path = status_data.get("file_path")
                    print(f"Found podcast status in file system: {status}")
            except Exception as e:
                print(f"Error reading status file: {e}")
                # Continue to database check if status file read fails
                status_data = None
        else:
            status_data = None
        
        # If no status file, check the database
        if not status_data:
            try:
                conn = sqlite3.connect(DB_PATH, timeout=10.0)
                cursor = conn.cursor()
                
                # Enable WAL mode for better concurrency
                cursor.execute("PRAGMA journal_mode=WAL;")
                
                cursor.execute(
                    "SELECT status, processed_path, processing_notes, created_at FROM injections WHERE id = ?", 
                    (injection_id,)
                )
                result = cursor.fetchone()
                conn.close()
                
                if not result:
                    return Title("Podcast Not Found"), Main(
                        Div(
                            H1("Podcast Not Found", cls="text-2xl font-bold text-center text-error mb-4"),
                            P(f"No podcast with ID: {injection_id} was found", cls="text-center mb-4"),
                            A("← Back to Home", href="/", cls="btn btn-primary block mx-auto"),
                            cls="container mx-auto px-4 py-8 max-w-3xl"
                        )
                    )
                
                status, audio_path, notes, created_at = result
            except Exception as e:
                print(f"Error querying database: {e}")
                # If both status file and database fail, show not found
                return Title("Podcast Not Found"), Main(
                    Div(
                        H1("Database Error", cls="text-2xl font-bold text-center text-error mb-4"),
                        P(f"Could not retrieve podcast with ID: {injection_id}", cls="text-center mb-4"),
                        P(f"Error: {str(e)}", cls="text-center mb-2 text-error"),
                        A("← Back to Home", href="/", cls="btn btn-primary block mx-auto"),
                        cls="container mx-auto px-4 py-8 max-w-3xl"
                    )
                )
        
        is_completed = status == "completed"
        
        # Check for the audio file with improved logic
        audio_exists = False
        file_path = None
        
        if is_completed:
            # Use our comprehensive find_audio_file function
            file_path = find_audio_file(injection_id)
            audio_exists = file_path is not None
            
            # If file found, update both status systems with the correct path
            if audio_exists:
                try:
                    update_injection_status(injection_id, "completed", f"Audio file found at {file_path}", max_retries=2)
                except Exception as e:
                    print(f"Error updating database with found path: {e}")
            
        # If we're receiving a request and status is already completed with audio,
        # assume this is the post-reload page view where we want to show the player
        audio_player = None
        if is_completed and audio_exists:
            audio_player = Div(
                H2("Listen to Your Podcast", cls="text-lg font-bold mb-3 text-white text-center"),
                Div(
                    NotStr(f'<audio src="/audio-raw/{injection_id}" controls autoplay class="w-full rounded-lg shadow mb-4"></audio>'),
                    A("Download Podcast", 
                    href=f"/audio-raw/{injection_id}", 
                    download=f"podcast_{injection_id}.wav", 
                    cls="btn btn-secondary w-full"),
                    cls="bg-black bg-opacity-30 p-4 rounded-lg"
                ),
                id="audio-player",
                cls="mb-6"
            )
        
        # Create animation style 
        animation_style = Style("""
        .animated-bg {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            height: 100vh;
        }
        
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        """)
        
        # Calculate how long the process has been running for smart backoff
        import datetime
        
        # Convert created_at from string to datetime if needed
        if isinstance(created_at, str):
            try:
                created_at = datetime.datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                created_at = datetime.datetime.now() - datetime.timedelta(minutes=1)
        
        # Calculate elapsed time
        now = datetime.datetime.now()
        elapsed_seconds = (now - created_at).total_seconds()
        
        # Determine polling interval based on elapsed time
        # More frequent at the start, less frequent as time passes
        poll_interval = 3000  # 3 seconds (in milliseconds)
        
        if elapsed_seconds > 60:  # After 1 minute
            poll_interval = 5000  # 5 seconds
        if elapsed_seconds > 120:  # After 2 minutes
            poll_interval = 10000  # 10 seconds
        if elapsed_seconds > 300:  # After 5 minutes
            poll_interval = 15000  # 15 seconds
        
        # Enhanced polling script with better error handling and audio detection
        # Enhanced polling script with better error handling and audio detection
        status_polling_script = Script(f"""
        // Global variable to hold the timeout ID
        let pollTimeoutId = null;
        let failedPollCount = 0;  // Added this declaration
        const MAX_FAILED_POLLS = 10;
                
        // Function to check status
        function checkStatus() {{
            console.log("Checking podcast status...");
            
            fetch('/podcast-status-api/{injection_id}')
                .then(response => response.json())
                .then(data => {{
                    console.log('Status update:', data);
                    failedPollCount = 0; // Reset failed poll counter
                    
                    // Update status display
                    const statusText = document.getElementById('status-text');
                    if (statusText) statusText.innerText = 'Status: ' + data.status;
                    
                    const statusNotes = document.getElementById('status-notes');
                    if (statusNotes) statusNotes.innerText = data.notes || 'Processing...';
                    
                    // Calculate progress percentage
                    let createdAt = data.timestamp;
                    if (typeof createdAt === 'string') {{
                        createdAt = new Date(createdAt).getTime();
                    }} else {{
                        createdAt = new Date().getTime() - 60000; // Fallback: 1 minute ago
                    }}
                    
                    const elapsed = (Date.now() - createdAt) / 1000; // seconds
                    const totalEstimated = 7 * 60; // 7 minutes in seconds
                    const progressPct = Math.min(95, Math.floor((elapsed / totalEstimated) * 100));
                    
                    // Update progress bar
                    const progressBar = document.getElementById('progress-bar');
                    if (progressBar) progressBar.style.width = progressPct + '%';
                    
                    // Update time estimate
                    const elapsedMin = Math.floor(elapsed / 60);
                    const elapsedSec = Math.floor(elapsed % 60);
                    const remainingMin = Math.max(0, Math.floor((totalEstimated - elapsed) / 60));
                    const remainingSec = Math.max(0, Math.floor((totalEstimated - elapsed) % 60));
                    
                    const timeInfo = document.getElementById('time-estimate');
                    if (timeInfo) {{
                        timeInfo.innerText = `Elapsed: ${{elapsedMin}}m ${{elapsedSec}}s | Est. remaining: ${{remainingMin}}m ${{remainingSec}}s`;
                    }}
                    
                    // IMPORTANT: If podcast is complete and audio exists, reload the page and stop polling
                    if (data.is_completed && data.audio_exists) {{
                        console.log('Podcast is ready! Stopping polls and reloading page...');
                        
                        // Show completion message
                        const statusIcon = document.getElementById('status-icon');
                        if (statusIcon) statusIcon.innerHTML = '<span class="text-4xl">✓</span>';
                        
                        if (statusNotes) statusNotes.innerText = 'Podcast is ready! Loading player...';
                        
                        // Very important: Clear the timeout to stop polling
                        if (pollTimeoutId) {{
                            console.log('Clearing timeout ID:', pollTimeoutId);
                            clearTimeout(pollTimeoutId);
                            pollTimeoutId = null;
                        }}
                        
                        // Brief delay then reload
                        setTimeout(() => window.location.reload(), 1000);
                        return; // Exit function immediately
                    }}
                    
                    // Handle audio-ready-but-not-marked-complete edge case
                    if (data.audio_exists && !data.is_completed) {{
                        console.log('Audio exists but status not marked complete. Reloading to check...');
                        // Clear timeout
                        if (pollTimeoutId) {{
                            clearTimeout(pollTimeoutId);
                            pollTimeoutId = null;
                        }}
                        // Brief delay then reload
                        setTimeout(() => window.location.reload(), 1000);
                        return;
                    }}
                    
                    // IMPORTANT: Only continue polling if not completed
                    if (!data.is_completed) {{
                        console.log('Podcast not complete, continuing to poll...');
                        pollTimeoutId = setTimeout(checkStatus, {poll_interval});
                    }} else {{
                        // If completed but no audio yet, poll at reduced frequency
                        console.log('Podcast completed but audio not ready, polling less frequently...');
                        pollTimeoutId = setTimeout(checkStatus, {poll_interval * 2});
                    }}
                }})
                .catch(error => {{
                    console.error('Error checking status:', error);
                    failedPollCount++;  // Make sure this is after variable is declared
                    console.log(`Poll attempt failed (${{failedPollCount}}/${{MAX_FAILED_POLLS}})`);
                    
                    // If too many failures, show error message
                    if (failedPollCount >= MAX_FAILED_POLLS) {{
                        const statusNotes = document.getElementById('status-notes');
                        if (statusNotes) statusNotes.innerText = 'Connection issues. Please refresh the page.';
                        if (pollTimeoutId) {{
                            clearTimeout(pollTimeoutId);
                            pollTimeoutId = null;
                        }}
                        return;
                    }}
                    
                    // Continue polling even on error, with longer interval
                    pollTimeoutId = setTimeout(checkStatus, {poll_interval * 2});
                }});
        }}
                
        // Start polling (only if we're not already on a page with a completed podcast)
        if (!document.getElementById('completed-indicator')) {{
            console.log('Starting status polling...');
            // Start the first poll
            checkStatus();
            
            // Cleanup on page unload
            window.addEventListener('beforeunload', function() {{
                if (pollTimeoutId) {{
                    console.log('Page unloading, clearing timeout');
                    clearTimeout(pollTimeoutId);
                }}
            }});
        }} else {{
            console.log('Podcast already completed, not starting polling');
        }}
        """) if not (is_completed and audio_exists) else None
        
        # Status indicator - different based on completed state
        status_indicator = Div(
            # Status icon (checkmark or loading indicator)
            Div(
                Span("✓", cls="text-4xl") if (is_completed and audio_exists) else NotStr('<div class="loading loading-dots loading-lg"></div>'),
                cls="text-success mb-6 text-center",
                id="status-icon"
            ),
            
            # Status text
            P(f"Status: {status}", cls="text-lg mb-2 text-center text-white", id="status-text"),
            P(notes or "Processing in progress...", cls="text-sm mb-4 text-center text-white", id="status-notes"),
            
            # Progress bar and time estimate (only for in-progress)
            Div(
                Div(
                    Div(
                        cls="bg-blue-400 h-2.5 rounded-full", 
                        style="width: 0%",
                        id="progress-bar"
                    ),
                    cls="w-full bg-gray-700 rounded-full h-2.5 mb-2"
                ),
                P(
                    "Calculating time estimate...",
                    cls="text-xs text-center text-white",
                    id="time-estimate"
                ),
                cls="mt-4"
            ) if not (is_completed and audio_exists) else None,
            
            # Add a completed indicator div if complete
            Div(id="completed-indicator") if (is_completed and audio_exists) else None,
            
            id="status-section",
            cls="p-6 bg-black bg-opacity-20 rounded-lg mb-6"
        )
        
        # Expected time info - only show if not completed with audio
        time_info = Div(
            P("Podcast generation can take 30-40 minutes.", 
            cls="text-center text-white mb-2"),
            P("This page will automatically update when your podcast is ready. But keep note of your Podcast ID!", 
            cls="text-center text-white mb-4"),
            cls="mb-4 p-4 bg-black bg-opacity-30 rounded-lg",
            id="time-info"
        ) if not (is_completed and audio_exists) else None
        
        # Main page content
        return Title("Podcast Status"), Main(
            animation_style,
            status_polling_script,
            Div(
                H1(
                    "Your Podcast is Ready!" if (is_completed and audio_exists) else "Your Podcast is Being Generated",
                    cls="text-3xl font-bold text-center text-white mb-6"
                ),
                status_indicator,
                time_info,
                audio_player,  # Only included if status is completed and audio exists
                A("← Back to Home", href="/", cls="btn btn-primary block mx-auto mt-6"),
                P(f"Podcast ID: {injection_id}", 
                cls="text-center text-white opacity-70 text-sm mt-4 font-mono"),
                cls="container mx-auto px-4 py-8 max-w-3xl"
            ),
            cls="animated-bg min-h-screen"
        )
    
    #################################################
    # Audio File Serving Endpoint
    #################################################
    @rt("/audio-raw/{injection_id}")
    def serve_audio_raw(injection_id: str):
        """Direct audio file handler with extensive fallbacks and retries"""
        print(f"📢 Audio request received for ID: {injection_id}")
        
        # Multiple explicit reload attempts
        for reload_attempt in range(3):
            shared_volume.reload()
            
            # Use the comprehensive find_audio_file function
            audio_path = find_audio_file(injection_id)
            
            # If we found a file, serve it right away
            if audio_path and os.path.exists(audio_path):
                print(f"✅ Serving audio file: {audio_path}")
                return FileResponse(
                    audio_path, 
                    media_type="audio/wav",
                    filename=f"podcast_{injection_id}.wav"
                )
            
            # Brief pause before next attempt
            time.sleep(0.5)
        
        # If all reload attempts failed, return error
        return HTMLResponse(
            """
            <div style="padding: 20px; background-color: #fff3cd; color: #856404; 
                 border-radius: 5px; margin: 20px auto; max-width: 600px; text-align: center;">
                <h3>Audio file not yet available</h3>
                <p>The podcast may still be processing. Please check back in a few minutes.</p>
                <p>If the issue persists, try refreshing the main status page.</p>
                <p><a href="/" style="color: #0056b3; text-decoration: underline;">Return to Homepage</a></p>
            </div>
            """,
            status_code=404
        )
    
    #################################################
    # Status Redirect Helper
    #################################################
    @rt("/status-redirect")
    def status_redirect(injection_id: str):
        """Redirect to the podcast status page"""
        return RedirectResponse(f"/podcast-status/{injection_id}")

    # Return the FastHTML app
    return fasthtml_app
