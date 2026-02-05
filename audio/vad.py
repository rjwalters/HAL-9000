"""
Voice Activity Detection using Silero VAD.
"""

import asyncio
import logging
import time
from collections import deque
from typing import Callable

import numpy as np
import torch

import config

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """Silero VAD wrapper for end-of-speech detection."""

    def __init__(
        self,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
        threshold: float = config.VAD_THRESHOLD,
        min_speech_ms: int = config.VAD_MIN_SPEECH_MS,
        min_silence_ms: int = config.VAD_MIN_SILENCE_MS,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms

        # Load Silero VAD model
        self._model = None
        self._load_model()

        # State tracking
        self._is_speaking = False
        self._speech_start_time: float | None = None
        self._silence_start_time: float | None = None

        # Audio buffer for VAD (needs 512 samples at 16kHz)
        self._audio_buffer: deque[float] = deque(maxlen=512)

        # Callbacks
        self._on_speech_start: Callable[[], None] | None = None
        self._on_speech_end: Callable[[], None] | None = None

    def _load_model(self) -> None:
        """Load Silero VAD model."""
        logger.info("Loading Silero VAD model...")
        try:
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            self._model.eval()
            logger.info("Silero VAD model loaded")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise

    def reset(self) -> None:
        """Reset VAD state."""
        self._is_speaking = False
        self._speech_start_time = None
        self._silence_start_time = None
        self._audio_buffer.clear()
        if self._model is not None:
            self._model.reset_states()

    def process_audio(self, audio_bytes: bytes) -> float:
        """
        Process audio chunk and return speech probability.

        Args:
            audio_bytes: Raw 16-bit PCM audio bytes

        Returns:
            Speech probability (0-1)
        """
        # Convert bytes to float32
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Add to buffer
        self._audio_buffer.extend(audio_float.tolist())

        # Need 512 samples for Silero VAD
        if len(self._audio_buffer) < 512:
            return 0.0

        # Get exactly 512 samples
        samples = list(self._audio_buffer)[-512:]
        audio_tensor = torch.tensor(samples, dtype=torch.float32)

        # Run VAD
        with torch.no_grad():
            speech_prob = self._model(audio_tensor, self.sample_rate).item()

        # Update state
        self._update_state(speech_prob)

        return speech_prob

    def _update_state(self, speech_prob: float) -> None:
        """Update speaking state based on speech probability."""
        current_time = time.time()
        is_speech = speech_prob >= self.threshold

        if is_speech:
            self._silence_start_time = None

            if not self._is_speaking:
                if self._speech_start_time is None:
                    self._speech_start_time = current_time
                elif (current_time - self._speech_start_time) * 1000 >= self.min_speech_ms:
                    # Speech confirmed
                    self._is_speaking = True
                    logger.debug("Speech started")
                    if self._on_speech_start:
                        self._on_speech_start()
        else:
            self._speech_start_time = None

            if self._is_speaking:
                if self._silence_start_time is None:
                    self._silence_start_time = current_time
                elif (current_time - self._silence_start_time) * 1000 >= self.min_silence_ms:
                    # End of speech confirmed
                    self._is_speaking = False
                    self._silence_start_time = None
                    logger.debug("Speech ended")
                    if self._on_speech_end:
                        self._on_speech_end()

    def on_speech_start(self, callback: Callable[[], None]) -> None:
        """Register callback for speech start."""
        self._on_speech_start = callback

    def on_speech_end(self, callback: Callable[[], None]) -> None:
        """Register callback for end of speech."""
        self._on_speech_end = callback

    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._is_speaking

    async def process_stream(
        self,
        audio_stream,
        on_speech_end: Callable[[], None] | None = None,
    ) -> None:
        """
        Process an async audio stream and detect speech events.

        Args:
            audio_stream: Async iterator yielding audio bytes
            on_speech_end: Optional callback when speech ends
        """
        if on_speech_end:
            self.on_speech_end(on_speech_end)

        async for audio_chunk in audio_stream:
            self.process_audio(audio_chunk)
            await asyncio.sleep(0)  # Yield to event loop
