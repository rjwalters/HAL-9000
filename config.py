"""
HAL-9000 Voice Assistant Configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# API Keys
# =============================================================================
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# =============================================================================
# Audio Configuration
# =============================================================================
AUDIO_SAMPLE_RATE = 16000  # Hz - standard for speech recognition
AUDIO_CHANNELS = 1  # Mono
AUDIO_CHUNK_SIZE = 512  # Samples per chunk (~32ms at 16kHz)
AUDIO_FORMAT_WIDTH = 2  # 16-bit audio (2 bytes)

# Playback configuration (ElevenLabs outputs 24kHz by default)
PLAYBACK_SAMPLE_RATE = 24000
PLAYBACK_CHANNELS = 1
PLAYBACK_CHUNK_SIZE = 1024

# =============================================================================
# VAD (Voice Activity Detection) Configuration
# =============================================================================
VAD_THRESHOLD = 0.5  # Speech probability threshold (0-1)
VAD_MIN_SPEECH_MS = 250  # Minimum speech duration to trigger
VAD_MIN_SILENCE_MS = 700  # Silence duration to mark end-of-speech
VAD_WINDOW_SIZE_MS = 32  # VAD analysis window

# =============================================================================
# Video Configuration
# =============================================================================
VIDEO_DEVICE_ID = 0  # Default webcam
VIDEO_FRAME_WIDTH = 640
VIDEO_FRAME_HEIGHT = 480
VIDEO_FPS = 30
VISION_CAPTURE_INTERVAL = 2.0  # Seconds between vision API calls
VISION_JPEG_QUALITY = 85  # JPEG compression quality (0-100)

# =============================================================================
# Deepgram STT Configuration
# =============================================================================
DEEPGRAM_MODEL = "nova-2"  # Latest model with best accuracy
DEEPGRAM_LANGUAGE = "en-US"
DEEPGRAM_SMART_FORMAT = True  # Auto-punctuation and formatting
DEEPGRAM_INTERIM_RESULTS = True  # Get partial results while speaking

# =============================================================================
# Groq LLM Configuration
# =============================================================================
GROQ_MODEL = "llama-3.3-70b-versatile"  # Llama 3.3 70B
GROQ_MAX_TOKENS = 256  # Keep responses concise for low latency
GROQ_TEMPERATURE = 0.7  # Slight creativity for HAL personality
GROQ_TOP_P = 0.9

# =============================================================================
# ElevenLabs TTS Configuration
# =============================================================================
ELEVENLABS_MODEL = "eleven_turbo_v2_5"  # Lowest latency model
ELEVENLABS_STABILITY = 0.7  # Voice consistency
ELEVENLABS_SIMILARITY_BOOST = 0.8  # Voice similarity to original
ELEVENLABS_STYLE = 0.0  # No style exaggeration for HAL's calm tone
ELEVENLABS_OPTIMIZE_LATENCY = 4  # Maximum latency optimization

# =============================================================================
# Claude Vision Configuration
# =============================================================================
VISION_MODEL = "claude-sonnet-4-20250514"  # Fast vision model
VISION_MAX_TOKENS = 150  # Brief scene descriptions

# =============================================================================
# HAL Eye Display Configuration
# =============================================================================
DISPLAY_HOST = "0.0.0.0"
DISPLAY_PORT = 8080
DISPLAY_WS_PATH = "/ws"

# =============================================================================
# State Machine
# =============================================================================
class State:
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
