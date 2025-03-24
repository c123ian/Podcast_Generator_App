import modal
import torch
import os
import ast
import pickle
import base64
from typing import Optional, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import Accelerator
from common_image import common_image, shared_volume
import re
import nltk
import textwrap

# Import sumy for dedicated summarization
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Setup NLTK for sentence splitting and summarization
NLTK_DATA_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

# Pre-download required NLTK resources to avoid issues during processing
try:
    nltk.download("punkt", download_dir=NLTK_DATA_DIR)
    nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)
    print("✅ NLTK resources downloaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Failed to download NLTK resources: {e}")

# Define app
app = modal.App("scripts_gen")

# Constants
DATA_DIR = "/data"
SCRIPTS_FOLDER = "/data/podcast_scripts"
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_INPUT_CHARS = 75000  # Consistent with input_gen.py
SUMMARIZATION_THRESHOLD = 30000  # Only summarize texts longer than this
TARGET_SUMMARY_LENGTH = 5000  # Target length for summarized content
MAX_SCRIPT_EXCHANGES = 38  # Maximum number of exchanges in a script

# Create necessary directories
os.makedirs(SCRIPTS_FOLDER, exist_ok=True)

# System prompts for script generation
SYSTEM_PROMPT = """
You are a world-class podcast writer, having ghostwritten for top shows like Joe Rogan, Lex Fridman, and Tim Ferris.
Your job is to write a lively, engaging script with two speakers based on the text I provide.
Speaker 1 leads the conversation, teaching Speaker 2, giving anecdotes and analogies.
Speaker 2 asks follow-up questions, gets excited or confused, and interrupts with umm, hmm occasionally.

IMPORTANT LENGTH CONSTRAINTS:
- Create a podcast script with EXACTLY 12-15 exchanges between speakers.
- The entire podcast should be about 5-7 minutes when read aloud.
- Keep the conversation focused and concise while maintaining engagement.
- If the source text is very long, focus on the most important and interesting aspects.

ALWAYS START YOUR RESPONSE WITH 'SPEAKER 1' and a colon.
PLEASE DO NOT GIVE OR MENTION THE SPEAKERS BY NAME.
Keep the conversation extremely engaging, welcome the audience with a fun overview, etc.
Only create ONE EPISODE of the podcast. 
The speakers discussing the topic as external commentators.
"""

REWRITE_PROMPT = """
You are an Oscar-winning screenwriter rewriting a transcript for an AI Text-To-Speech Pipeline.

Make it as engaging as possible, Speaker 1 and 2 will be using different voices.

IMPORTANT LENGTH CONSTRAINTS:
- The final script should be EXACTLY 12-15 exchanges between speakers.
- The entire podcast should be about 5-7 minutes when read aloud.

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

Please re-write to make it as characteristic as possible

For both Speakers use the following disfluencies FREQUENTLY AS MUCH AS POSSIBLE, umm, hmm, [laughs], [sighs], [laughter], [gasps], [clears throat], — for hesitations, CAPITALIZATION for emphasis. BUT ONLY THESE OPTIONS FOR EXPRESSIONS

Do not use [excitedly], [trailing off)], [interrupting], [pauses] or anything else that is NOT an outlined above disfluency for expression.

Return your final answer as a Python LIST of (Speaker, text) TUPLES ONLY, NO EXPLANATIONS, e.g.

Dont add "quotation marks" within the script dialogue. 

IT WILL START DIRECTLY WITH THE LIST AND END WITH THE LIST NOTHING ELSE

[
    ("Speaker 1", "Hello, and welcome..."),
    ("Speaker 2", "Hmm, that is fascinating!")
]

IMPORTANT Your response must be a valid Python list of tuples. STRICTLY RETURN YOUR RESPONSE AS A LIST OF TUPLES
"""

def clean_llm_output(text: str) -> str:
    """Remove prompt instructions from the LLM output"""
    # List of markers that indicate where the actual content starts
    content_markers = [
        'SPEAKER 1:', 
        'Speaker 1:',
        'Speaker 1,', 
        'SPEAKER 1 ,',  
        '[("Speaker 1"',
        '[("SPEAKER 1"'
    ]
    
    # Find the earliest occurrence of any marker
    start_index = float('inf')
    for marker in content_markers:
        idx = text.find(marker)
        if idx != -1 and idx < start_index:
            start_index = idx
    
    if start_index == float('inf'):
        return text
    
    return text[start_index:]


def convert_disfluencies(text):
    """
    Convert parenthesized expressions like (laughs) to bracketed [laughs]
    for proper TTS rendering.
    """
    # List of common disfluencies to check for
    disfluencies = [
        "laughs", "sighs", "laughter", "gasps", "clears throat", 
        "sigh", "laugh", "gasp", "chuckles", "snorts",
        "hmm", "umm", "uh", "ah", "er", "um"
    ]
    
    # Convert (laughs) to [laughs]
    for disfluency in disfluencies:
        # Look for various formats and convert them
        text = re.sub(r'\((' + disfluency + r')\)', r'[\1]', text, flags=re.IGNORECASE)
        text = re.sub(r'<(' + disfluency + r')>', r'[\1]', text, flags=re.IGNORECASE)
        
        # Also match when there's text inside
        text = re.sub(r'\(([^)]*' + disfluency + r'[^)]*)\)', r'[\1]', text, flags=re.IGNORECASE)
        
    return text

def summarize_long_text(text, max_length=TARGET_SUMMARY_LENGTH):
    # Use a more aggressive approach - target much shorter output
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    
    # Target significantly fewer sentences - aim for 15-20% of original length
    target_sentences = max(15, int(max_length // 200))  # ~200 chars per sentence
    
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, target_sentences)
    
    # Join sentences into coherent text and truncate if still needed
    result = " ".join(str(sentence) for sentence in summary_sentences)
    if len(result) > max_length:
        result = result[:max_length-100] + "... [Content truncated due to length]"
    
    return result

def parse_podcast_script(text: str) -> List[Tuple[str, str]]:
    """Parse the generated text into a list of speaker-text tuples"""
    # Clean the output first
    cleaned_text = clean_llm_output(text)
    
    try:
        # Extract everything between the first [ and last ]
        start_idx = cleaned_text.find("[")
        end_idx = cleaned_text.rfind("]") + 1
        if start_idx == -1 or end_idx == 0:
            # If no brackets found, try to parse line by line
            lines = cleaned_text.split('\n')
            script = []
            current_speaker = None
            current_text = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if ':' in line:
                    if current_speaker and current_text:
                        script.append((current_speaker, ' '.join(current_text)))
                        current_text = []
                    
                    parts = line.split(':', 1)
                    current_speaker = parts[0].strip()
                    if len(parts) > 1:
                        current_text.append(parts[1].strip())
                else:
                    if current_speaker:
                        current_text.append(line)
            
            if current_speaker and current_text:
                script.append((current_speaker, ' '.join(current_text)))
            
            return script if script else [("Speaker 1", cleaned_text)]
            
        # Try to parse the content between brackets
        candidate = cleaned_text[start_idx:end_idx]
        final_script = ast.literal_eval(candidate)
        
        if not isinstance(final_script, list):
            return [("Speaker 1", cleaned_text)]
            
        return final_script
        
    except Exception as e:
        print(f"Parsing error: {str(e)}")
        # Fallback to basic parsing if ast.literal_eval fails
        try:
            lines = cleaned_text.split('\n')
            script = []
            for line in lines:
                if ':' in line:
                    parts = line.split(':', 1)
                    speaker = parts[0].strip()
                    text = parts[1].strip() if len(parts) > 1 else ""
                    script.append((speaker, text))
            return script if script else [("Speaker 1", cleaned_text)]
        except Exception as e2:
            print(f"Fallback parsing error: {str(e2)}")
            return [("Speaker 1", cleaned_text)]


def normalize_script_quotes(script):
    """Normalize the entire script format to match the expected output."""
    normalized_script = []
    
    for speaker, text in script:
        # 1. Standardize speaker format (Speaker 1, Speaker 2)
        if speaker.upper() == "SPEAKER 1":
            speaker = "Speaker 1"
        elif speaker.upper() == "SPEAKER 2":
            speaker = "Speaker 2"
        
        # 2. Standardize all quotes - more thorough approach
        # First, protect contractions
        text = re.sub(r'(\w)\'(\w)', r'\1APOSTROPHE\2', text)
        
        # Replace all quotes (single and double) with double quotes
        text = text.replace('"', '"').replace("'", '"')
        
        # Restore apostrophes in contractions
        text = text.replace('APOSTROPHE', "'")
        
        # 3. Convert disfluencies format
        text = convert_disfluencies(text)
        
        normalized_script.append((speaker, text))
    
    return normalized_script

# Additional helper function
def truncate_script(script_data, max_exchanges=MAX_SCRIPT_EXCHANGES):
    """Ensure script doesn't exceed maximum number of exchanges"""
    if len(script_data) <= max_exchanges:
        return script_data
        
    print(f"⚠️ Script exceeds maximum exchanges ({len(script_data)} > {max_exchanges}), truncating...")
    
    # Ensure we have an even number of exchanges and a clean ending
    end_idx = max_exchanges
    if max_exchanges % 2 == 1:  # If odd number, make it even
        end_idx = max_exchanges - 1
    
    # Make sure the last exchange is from Speaker 2 for a natural ending
    if end_idx >= 2:
        speaker_last = script_data[end_idx-1][0]
        speaker_second_last = script_data[end_idx-2][0]
        
        if speaker_last == speaker_second_last:
            end_idx = end_idx - 1
    
    return script_data[:end_idx]

# NEW HELPER FUNCTION TO HANDLE THE "---" MARKER ISSUE
def truncate_at_separator(script_data):
    """
    Truncate script at any "---" separator markers, which indicate the end of usable content.
    This fixes issues with malformed scripts that have content after separators.
    """
    truncated_script = []
    
    for speaker, text in script_data:
        # Check if this text contains the separator
        if " --- " in text:
            # Keep only the content before the first separator
            clean_text = text.split(" --- ")[0].strip()
            print(f"⚠️ Found separator marker in script. Truncating text from {len(text)} to {len(clean_text)} chars")
            truncated_script.append((speaker, clean_text))
            # Stop processing after finding a separator
            break
        else:
            # Keep the line as is
            truncated_script.append((speaker, text))
    
    # If we truncated anything, add a clean ending if needed
    if len(truncated_script) < len(script_data):
        # If the last line is from Speaker 1, add a closing line from Speaker 2
        if truncated_script and truncated_script[-1][0] == "Speaker 1":
            truncated_script.append(("Speaker 2", "That's really fascinating! Thank you for explaining that to me."))
            
    return truncated_script if truncated_script else script_data

# Load Llama8B specific image
llama_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "transformers==4.46.1",
        "accelerate>=0.26.0",
        "torch>=2.0.0",
        "sentencepiece",
        "sumy>=0.11.0",  # Add sumy for summarization
        "nltk>=3.8.1"    # Required by sumy
    )
)

# Look up Llama model volume
try:
    llm_volume = modal.Volume.lookup("llamas_8b", create_if_missing=False)
    LLAMA_DIR = "/llamas_8b"
    print("Llama 8B model volume found")
except modal.exception.NotFoundError:
    # Fall back to using the common image if Llama model not available
    llm_volume = None
    LLAMA_DIR = None
    print("Warning: 'llamas_8b' volume not found. Will use OpenAI API fallback if configured.")

@app.function(
    image=llama_image if LLAMA_DIR else common_image,
    gpu=modal.gpu.A10G(count=1),
    container_idle_timeout=10 * 60,
    timeout=24 * 60 * 60,
    volumes={LLAMA_DIR: llm_volume, "/data": shared_volume} if LLAMA_DIR else {"/data": shared_volume},
)
def generate_script(source_text: str, retry_count: int = 0) -> str:
    """
    Generates a podcast script from source text with dedicated summarization
    but preserving the two-step generation process.
    Returns the base64 encoded pickled script (list of tuples).
    """
    print(f"🚀 Generating script from text of length: {len(source_text)} characters")
    
    # If this is a retry, log it
    if retry_count > 0:
        print(f"⚠️ This is retry attempt #{retry_count} after detecting problematic script size")
    
    # Step 1: Use dedicated summarization for very long texts
    if len(source_text) > SUMMARIZATION_THRESHOLD:
        try:
            # No need to download NLTK data here - already done at module level
            source_text = summarize_long_text(source_text)
        except Exception as e:
            print(f"⚠️ Error during summarization: {e}. Using basic truncation instead.")
            # Fallback to simple truncation if summarization fails
            source_text = source_text[:TARGET_SUMMARY_LENGTH]
            print(f"Truncated text to {len(source_text)} characters")
    
    if LLAMA_DIR:
        # Use local Llama model
        print("Loading Llama model...")
        accelerator = Accelerator()
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_DIR,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_DIR)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
            
        model, tokenizer = accelerator.prepare(model, tokenizer)
        
        generation_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            max_new_tokens=1500,
            temperature=0.7,      # Slightly lower temperature for more focused output
        )

        # Step 2: Generate initial script with first prompt
        print("Generating initial script...")
        prompt_1 = SYSTEM_PROMPT + "\n\n" + source_text
        first_draft = generation_pipe(prompt_1)[0]["generated_text"]

        # Step 3: Rewrite with disfluencies using second prompt
        print("Adding natural speech patterns...")
        prompt_2 = REWRITE_PROMPT + "\n\n" + first_draft
        final_text = generation_pipe(prompt_2)[0]["generated_text"]
        
        # Step 4: Parse and normalize the script
        script_data = parse_podcast_script(final_text)
        
        print(f"Raw script has {len(script_data)} exchanges")
        
        # Step 5: Ensure script doesn't exceed maximum length
        script_data = truncate_script(script_data)

    else:
        # Add OpenAI fallback here if needed
        print("Using OpenAI API fallback (not implemented)")
        # This would be where you add OpenAI API call code
        final_text = f"""
        [
            ("Speaker 1", "Using OpenAI API fallback Welcome to our podcast! Today we're discussing an interesting topic from the provided content."),
            ("Speaker 2", "Hmm, sounds fascinating! What's it about?"),
            ("Speaker 1", "It's about {source_text[:100]}..."),
            ("Speaker 2", "Tell me more about that!")
        ]
        """
        script_data = parse_podcast_script(final_text)

    # Step 6: Normalize quotes and disfluencies
    final_script = normalize_script_quotes(script_data)

    # Step 7: Final format validation
    for i, (speaker, text) in enumerate(final_script):
        # Final validation of format
        if not (speaker == "Speaker 1" or speaker == "Speaker 2"):
            final_script[i] = (f"Speaker {1 if '1' in speaker else 2}", text)

    print(f"Final script format sample: {final_script[0] if final_script else '(empty)'}")
    print(f"Final script length: {len(final_script)} exchanges")
            
    # Save the script to a file for debugging/reference
    import uuid
    file_uuid = uuid.uuid4().hex
    pkl_path = os.path.join(SCRIPTS_FOLDER, f"script_{file_uuid}.pkl")
    
    with open(pkl_path, "wb") as f:
        pickle.dump(final_script, f)
    
    print(f"Script saved to {pkl_path} with {len(final_script)} lines of dialogue")
    
    # Serialize the script for passing to the audio generation function
    serialized_data = pickle.dumps(final_script)
    
    # NEW CODE: Check if the size of the serialized data is too large (≥ 10.1 KiB)
    max_size_kb = 10.1 * 1024  # 10.1 KiB in bytes as the threshold
    actual_size = len(serialized_data)
    actual_size_kb = actual_size / 1024
    
    # Check if size is at or above the problematic threshold
    is_too_large = actual_size >= max_size_kb
    max_retries = 3
    
    if is_too_large:
        print(f"⚠️ Detected large script size: {actual_size} bytes ({actual_size_kb:.2f} KiB) - threshold is {max_size_kb/1024:.2f} KiB")
        
        if retry_count < max_retries:
            print(f"⚠️ This large size may cause issues. Restarting process to generate a smaller script (attempt {retry_count + 1}/{max_retries})...")
            
            # Add a directive to generate a more concise script
            random_seed = str(uuid.uuid4().hex)[:8]  # Use a random seed for variation
            modified_text = source_text + f"\n\nImportant: Generate a CONCISE podcast script with fewer exchanges. Keep it brief. [seed:{random_seed}]"
            
            # KEY FIX: Use .remote() instead of directly calling the function recursively
            return generate_script.remote(modified_text, retry_count + 1)
        else:
            print(f"⚠️ Still generated large scripts after {max_retries} attempts. Creating compact fallback script...")
            
            # Create a guaranteed fallback script that's smaller
            fallback_script = [
                ("Speaker 1", f"Welcome to our podcast! Today we're exploring the topic: '{source_text[:50]}...'"),
                ("Speaker 2", "Hmm, sounds interesting! What are the main points?"),
                ("Speaker 1", f"The key aspects are as follows. First, {source_text[50:150]}..."),
                ("Speaker 2", "I see. And what's the significance of this?"),
                ("Speaker 1", "That's a great question. In essence..."),
                ("Speaker 2", "Any final takeaways for our listeners?"),
                ("Speaker 1", "Absolutely. To summarize: understand the context, analyze the implications, and apply the insights. Thanks for listening!")
            ]
            
            # Replace our large script with the compact fallback
            serialized_data = pickle.dumps(fallback_script)
            actual_size = len(serialized_data)
            actual_size_kb = actual_size / 1024
            print(f"Created compact fallback script with size: {actual_size_kb:.2f} KiB")
    
    encoded_data = base64.b64encode(serialized_data).decode('utf-8')
    
    # Log final script info
    print(f"Final serialized script size: {actual_size_kb:.2f} KiB")
    if retry_count > 0 and not is_too_large:
        print(f"✅ Successfully generated smaller script after {retry_count} retry attempts")
        
    return encoded_data
