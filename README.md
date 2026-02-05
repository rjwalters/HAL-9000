# HAL-9000 Voice Assistant

A low-latency voice assistant with HAL-9000's personality that can see and hear you. Built for interactive installations and parties.

```
"I'm sorry, Dave. I'm afraid I can't do that."
```

## Features

- **Voice Interaction**: Natural conversation with <2 second response latency
- **Computer Vision**: Sees and comments on its environment using Claude Vision
- **HAL Personality**: Menacing calm, formal speech, and subtle observations
- **Live Eye Display**: Web-based HAL eye that reacts to speech amplitude
- **Streaming Pipeline**: Sentences synthesized as LLM generates them

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator (main.py)               │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Microphone   │  │ Webcam       │  │ Speaker      │  │
│  │ (PyAudio)    │  │ (OpenCV)     │  │ (PyAudio)    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────▲───────┘  │
│         │                 │                 │          │
│         ▼                 ▼                 │          │
│  ┌──────────────┐  ┌──────────────┐         │          │
│  │ Silero VAD   │  │ Claude       │         │          │
│  │ (end-of-     │  │ Vision API   │         │          │
│  │  speech)     │  │ (scene desc) │         │          │
│  └──────┬───────┘  └──────┬───────┘         │          │
│         │                 │                 │          │
│         ▼                 │                 │          │
│  ┌──────────────┐         │                 │          │
│  │ Deepgram STT │         │                 │          │
│  │ (streaming)  │         │                 │          │
│  └──────┬───────┘         │                 │          │
│         │                 │                 │          │
│         └────────┬────────┘                 │          │
│                  ▼                          │          │
│           ┌──────────────┐                  │          │
│           │ Groq LLM     │                  │          │
│           │ Llama 3.3 70B│                  │          │
│           └──────┬───────┘                  │          │
│                  │                          │          │
│                  ▼                          │          │
│           ┌──────────────┐                  │          │
│           │ ElevenLabs   ├──────────────────┘          │
│           │ TTS Stream   │                             │
│           └──────────────┘                             │
└─────────────────────────────────────────────────────────┘
```

## Latency Budget

| Component | Target | Service |
|-----------|--------|---------|
| VAD + End-of-speech | 200ms | Silero VAD (local) |
| Speech-to-Text | 300ms | Deepgram (streaming) |
| Vision | 0ms* | Claude Vision (pre-cached) |
| LLM (TTFT) | 200ms | Groq (Llama 3.3 70B) |
| TTS (first audio) | 300ms | ElevenLabs (streaming) |
| **Total** | **~1.0s** | |

*Vision runs continuously in background, always has recent frame ready

## Setup

### Prerequisites

- Python 3.11+
- macOS (for PyAudio) or Linux
- Webcam and microphone
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/rjwalters/HAL-9000.git
cd HAL-9000

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# On macOS, you may need portaudio for PyAudio
brew install portaudio
```

### API Keys

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required keys:
- `DEEPGRAM_API_KEY` - [Deepgram](https://deepgram.com/) for speech-to-text
- `GROQ_API_KEY` - [Groq](https://groq.com/) for LLM inference
- `ELEVENLABS_API_KEY` - [ElevenLabs](https://elevenlabs.io/) for text-to-speech
- `ELEVENLABS_VOICE_ID` - Your cloned HAL voice ID
- `ANTHROPIC_API_KEY` - [Anthropic](https://anthropic.com/) for Claude Vision

### Voice Cloning (Optional)

For the authentic HAL-9000 experience, clone Douglas Rain's voice:

1. Get audio samples from 2001: A Space Odyssey
2. Use the included extraction tool:
   ```bash
   python tools/extract_hal_audio.py movie.mkv subtitles.srt -o ./hal_samples
   ```
3. Upload samples to ElevenLabs Voice Lab
4. Copy the voice ID to your `.env`

## Usage

```bash
# Activate virtual environment
source .venv/bin/activate

# Run HAL-9000
python main.py
```

Open http://localhost:8080 to see the HAL eye display.

### State Machine

HAL operates in four states:
- **IDLE**: Eye breathes slowly, waiting for speech
- **LISTENING**: Eye pulses, capturing user speech
- **PROCESSING**: Eye rotates, generating response
- **SPEAKING**: Eye glows with audio amplitude

## Project Structure

```
HAL-9000/
├── main.py                 # Orchestrator and entry point
├── config.py               # Configuration and API keys
├── requirements.txt        # Python dependencies
├── .env.example            # API keys template
│
├── audio/
│   ├── input.py            # Microphone capture
│   ├── output.py           # Speaker playback
│   └── vad.py              # Silero VAD wrapper
│
├── video/
│   └── capture.py          # Webcam capture
│
├── services/
│   ├── stt.py              # Deepgram streaming client
│   ├── llm.py              # Groq LLM client
│   ├── tts.py              # ElevenLabs TTS client
│   └── vision.py           # Claude Vision client
│
├── hal/
│   ├── personality.py      # HAL-9000 system prompt
│   └── conversation.py     # Conversation history
│
├── display/
│   ├── server.py           # FastAPI WebSocket server
│   └── static/
│       ├── index.html      # HAL eye display
│       └── eye.js          # Eye animation
│
└── tools/
    └── extract_hal_audio.py  # Audio sample extractor
```

## Configuration

Key settings in `config.py`:

```python
# Audio
AUDIO_SAMPLE_RATE = 16000      # Hz
VAD_MIN_SILENCE_MS = 700       # End-of-speech detection

# Vision
VISION_CAPTURE_INTERVAL = 2.0  # Seconds between vision updates

# LLM
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS = 256          # Keep responses concise

# Display
DISPLAY_PORT = 8080
```

## Troubleshooting

**Camera not working on macOS**
- Grant camera permission in System Settings > Privacy & Security > Camera

**PyAudio installation fails**
- Install portaudio first: `brew install portaudio`

**TTS returns 402 error**
- ElevenLabs library voices require a paid subscription
- Create your own voice clone (free tier) or upgrade

**Empty transcripts from STT**
- Check microphone permissions
- Adjust `VAD_THRESHOLD` in config.py
- Ensure clear speech input

## License

MIT

## Acknowledgments

- HAL 9000 created by Stanley Kubrick and Arthur C. Clarke
- Douglas Rain provided HAL's iconic voice in 2001: A Space Odyssey
