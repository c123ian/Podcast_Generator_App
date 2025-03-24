# Podcast Generator

A system that automatically creates podcasts from any text content. Upload a PDF, text file, website, or audio, and the system generates a natural conversational podcast between two speakers using AI.

![image](https://github.com/user-attachments/assets/568c1d2f-8cde-4813-922e-f1cdc5e1d100)




## Samples

https://github.com/user-attachments/assets/24884ec4-8d88-43cb-9385-3144afe6f50b

https://github.com/user-attachments/assets/2f1f1d26-bb38-4f5b-8a2d-19943342318c

https://github.com/user-attachments/assets/c1c6a7e6-d735-4b34-9526-8481bd0693a9



## 🎙️ Overview

This application takes content from various sources, transforms it into a conversational podcast script, adds natural speech patterns (umms, laughs, pauses), and generates high-quality audio using Bark TTS. The entire pipeline runs on Modal Labs for scalable cloud deployment.



### ✨ Key Features

- **Multiple Input Types**: Process PDFs, web pages, text files, or audio transcriptions
- **Two-Speaker Conversational Format**: Content is rewritten as an engaging dialogue
- **Natural Speech Patterns**: Adds disfluencies like "umm", "[laughs]", pauses, and emphasis
- **Consistent Voice Models**: Maintains consistent voices across the entire podcast
- **Progress Tracking**: Shows real-time generation progress
- **Web UI**: Upload content and monitor podcast creation progress
- **Audio Playback**: Listen to the podcast directly in your browser

## 🏗️ Architecture

The system consists of three main components based on the [Llama cookbook example](https://github.com/meta-llama/llama-cookbook/tree/main/end-to-end-use-cases/NotebookLlama) that run as separate but [coordinated services](https://modal.com/docs/guide/project-structure):

![image](https://github.com/user-attachments/assets/b1428214-507f-4fef-a6ca-fcdb0389f41c)


### 📁 Files and Their Roles

1. **`deploy.py`**: Main orchestration file that combines all components
2. **`common_image.py`**: Defines shared resources (Docker image, volume)
3. **`input_gen.py`**: Web UI, file processing, and content extraction
4. **`scripts_gen.py`**: Converts raw content to podcast scripts using language models
5. **`audio_gen.py`**: Converts scripts to audio using Bark TTS

## 🚀 Installation

### Prerequisites

- Python 3.10+
- [Modal CLI](https://modal.com/docs/guide/cli-reference) installed and authenticated
- Modal account (free tier works for testing)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-file-podcast.git
   cd multi-file-podcast
   ```

2. Install dependencies:
   ```bash
   pip install modal python-fasthtml PyPDF2 langchain-community openai-whisper 
   ```

3. Create Modal volume for persistent storage:
   ```bash
   modal volume create combined_volume
   ```

4. Download LLM to Modal volume:
   ```bash
   modal run download_llama_8b.py
   ```

   ```bash
   modal run download_bark.py
   ```

5. Deploy to Modal:
   ```bash
   modal deploy deploy.py
   ```

## ⚙️ Technical Details

### Overcoming Bark's 13-Second Limitation

Bark TTS natively only supports ~13 seconds of audio generation per invocation. We implemented a sophisticated long-form generation pipeline based on the approach from [Bark's long-form notebook](https://github.com/suno-ai/bark/blob/main/notebooks/long_form_generation.ipynb), which involved:

```python
# Split text into sentences using NLTK tokenizer
sentences = nltk.sent_tokenize(preprocess_text(full_text))

# Process sentence by sentence while maintaining voice consistency
for sent in sentences:
    semantic_tokens = generate_text_semantic(
        sent,
        history_prompt=voice_preset,
        temp=text_temp,
        min_eos_p=0.05,  # Lower threshold prevents hallucinations
    )
    
    audio_array = semantic_to_waveform(
        semantic_tokens, 
        history_prompt=voice_preset,
        temp=waveform_temp,
    )
    
    all_audio.append(audio_array)
    all_audio.append(chunk_silence)  # Add consistent silence between sentences
```

### Lower-Level API for Enhanced Control

We abandoned the high-level `generate_audio()` API in favor of the lower-level functions:
- `generate_text_semantic()`: Converts text to semantic tokens
- `semantic_to_waveform()`: Renders semantic tokens to audio waveform

This gave us better control over:
- Temperature parameters (`text_temp` and `waveform_temp`) for both stages
- End-of-sentence detection via `min_eos_p` parameter
- Silence insertion between sentences

### Parallel Processing Architecture

We implemented a speaker-based parallelization strategy:
```python
# --- Split by speaker for parallel processing ---
speaker1_lines = [(i, text) for i, (speaker, text) in enumerate(lines) if speaker == "Speaker 1"]
speaker2_lines = [(i, text) for i, (speaker, text) in enumerate(lines) if speaker == "Speaker 2"]

# Process each speaker's lines in parallel on separate GPUs
if speaker1_lines:
    speaker1_results = generate_speaker_audio.remote(speaker1_lines, "Speaker 1", injection_id)

if speaker2_lines:
    speaker2_results = generate_speaker_audio.remote(speaker2_lines, "Speaker 2", injection_id)

# Recombine in original script order
all_results = speaker1_results + speaker2_results
all_results.sort(key=lambda x: x[0])  # Sort by original line index
```

This approach:
- Deploys two separate GPU containers simultaneously
- Maintains consistent voice characteristics per speaker
- I think cuts generation time roughly in half

### Voice Consistency Through Modal.Dict

Maintaining consistent voices across distributed containers was a significant challenge. We solved this with Modal's distributed dictionary:

```python
voice_states = modal.Dict.from_name("voice-states", create_if_missing=True)

# For each speaker, create a deterministic RNG seed
voice_state_key = f"{injection_id}_{speaker}"
if voice_states.contains(voice_state_key):
    seed = voice_states.get(voice_state_key)
else:
    # First-time generation creates a new seed (different per speaker)
    speaker_num = 1 if speaker == "Speaker 1" else 2
    seed = np.random.randint(10000 * speaker_num, 10000 * (speaker_num + 1) - 1)
    voice_states[voice_state_key] = seed
```

## Script Processing Optimizations

### Community-Discovered Disfluencies

The project leverages community-discovered Bark "hidden features" for disfluencies. These aren't officially documented, but were found through experimentation:

```python
def convert_disfluencies(text):
    """
    Convert parenthesized expressions like (laughs) to bracketed [laughs]
    for proper TTS rendering.
    """
    disfluencies = [
        "laughs", "sighs", "laughter", "gasps", "clears throat", 
        "sigh", "laugh", "gasp", "chuckles", "snorts",
        "hmm", "umm", "uh", "ah", "er", "um"
    ]
    
    for disfluency in disfluencies:
        # Convert various formats to [disfluency]
        text = re.sub(r'\(' + disfluency + r'\)', '[' + disfluency + ']', text, flags=re.IGNORECASE)
        text = re.sub(r'<' + disfluency + r'>', '[' + disfluency + ']', text, flags=re.IGNORECASE)
    
    return text
```

Not all disfluencies work equally well; they're emergent behaviors that Bark learned during training rather than explicitly programmed features. `[laughs]` and `hmm` tend to work reliably, while others are less consistent.

### Two-Pass Script Generation

Our script generation follows a sophisticated two-stage approach:

1. **Initial Script Generation**: Uses a system prompt focused on dialogue structure and content organization
   ```python
   prompt_1 = SYSTEM_PROMPT + "\n\n" + source_text
   first_draft = generation_pipe(prompt_1)[0]["generated_text"]
   ```

2. **Dramatic Rewriting**: A second-pass with a specialized prompt to add natural speech patterns
   ```python
   prompt_2 = REWRITE_PROMPT + "\n\n" + first_draft
   final_text = generation_pipe(prompt_2)[0]["generated_text"]
   ```

The rewrite prompt specifically instructs the model to add disfluencies:
```
REMEMBER THIS WITH YOUR HEART: Re-inject disfluencies FREQUENTLY 
but ONLY use the following at MINIMUM once: "umm, hmm, [laughs], 
[sighs], [laughter], [gasps], [clears throat], — for hesitations, 
CAPITALIZATION for emphasis of a word".
```

### Script Standardization for TTS

Script standardization was crucial for consistent audio generation:

```python
def normalize_script_quotes(script):
    """Normalize the entire script format to match the expected output."""
    normalized_script = []
    
    for speaker, text in script:
        # Standardize speaker format
        if speaker.upper() == "SPEAKER 1":
            speaker = "Speaker 1"
        elif speaker.upper() == "SPEAKER 2":
            speaker = "Speaker 2"
        
        # Protect contractions
        text = re.sub(r'(\w)\'(\w)', r'\1APOSTROPHE\2', text)
        
        # Standardize all quotes
        text = text.replace('"', '"').replace("'", '"')
        
        # Restore apostrophes in contractions
        text = text.replace('APOSTROPHE', "'")
        
        # Apply disfluencies conversion
        text = convert_disfluencies(text)
        
        normalized_script.append((speaker, text))
    
    return normalized_script
```

This standardization addresses a subtle but critical issue: Bark produces extended silence when encountering inconsistent quote formats. By normalizing to a single standard, we achieved much more fluid speech.


## 🔍 Further Development (TO DO)

- Experiment more with adjusting the prompt/upgrading LLM size, especially to addrress perspective issue:
   - Currently podcasters take a first-person perspective (i.e., speakers embody the CV subject).
   - Better to retain podcasters in the third-person perspective (speakers analyze the CV as external content), similar to Google NotebookLLM.
   - This shift from "persona adoption" to "content discussion" would create a more natural podcast format.
   - Also podcaster's drag out their goodbyes, they might need some encouragmenet to end the show a little more concisley via prompt! 
- Support additional [input formats](https://github.com/meta-llama/llama-recipes/pull/750) (i.e. YouTube URLs, requires Modal Labs 'Team 'subscription in order to avail of IP Proxy)
- Take advantage of Barks [MUSIC], implement music at the start or end of a podcast using Bark's `♪ ` surronding lyrics gnerated by an LLM (based on podcast topic).
   - The TTS sometimes does hallucinate/generate music at the end of a podcast by itself.
   - Use Bark-small would reduce generation time further but at a quality cost
- Experiment with an alternative TTS model, perhaps with faster generation, could try the newly released [1B CSM variant](https://github.com/SesameAILabs/csm)
- Add more granular control over podcast style and format (allow user to make addional comments and concatinate that to the scrip generating prompt - similar to Googl'es NotebookLLM feature).
- ~~Improve summarization for very long content rather then harsh truncation.~~
- ~~Upload bark model to a Modal Volume.~~
- Use [server-side-events (SSE)](https://github.com/DrChrisLevy/DrChrisLevy.github.io/blob/main/posts/sse/sse.ipynb) to tigger the audio player.


## 🙏 Acknowledgments

- [Modal Labs](https://modal.com) for serverless infrastructure
- [Bark TTS](https://github.com/suno-ai/bark) for high-quality text-to-speech
- [FastHTML](https://github.com/fastai/fasthtml) for the web UI framework
- [LangChain](https://github.com/langchain-ai/langchain) for document processing
- [llama-cookbook](https://github.com/meta-llama/llama-cookbook/tree/main/end-to-end-use-cases/NotebookLlama) for the NotebookLlama foundation code
