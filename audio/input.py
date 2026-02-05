"""
Audio input capture using PyAudio.
"""

import asyncio
import logging
from typing import AsyncIterator, Callable

import numpy as np
import pyaudio

import config

logger = logging.getLogger(__name__)


class AudioInput:
    """Microphone capture with async streaming."""

    def __init__(
        self,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
        channels: int = config.AUDIO_CHANNELS,
        chunk_size: int = config.AUDIO_CHUNK_SIZE,
        device_index: int | None = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index

        self._pyaudio: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None
        self._running = False
        self._callbacks: list[Callable[[bytes], None]] = []

    def start(self) -> None:
        """Start audio capture."""
        if self._running:
            return

        self._pyaudio = pyaudio.PyAudio()

        # Log available devices
        logger.info("Available audio input devices:")
        for i in range(self._pyaudio.get_device_count()):
            info = self._pyaudio.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                logger.info(f"  [{i}] {info['name']}")

        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback,
        )

        self._running = True
        logger.info(
            f"Audio input started: {self.sample_rate}Hz, "
            f"{self.channels}ch, chunk={self.chunk_size}"
        )

    def stop(self) -> None:
        """Stop audio capture."""
        self._running = False

        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None

        logger.info("Audio input stopped")

    def _audio_callback(
        self,
        in_data: bytes,
        frame_count: int,
        time_info: dict,
        status: int,
    ) -> tuple[None, int]:
        """PyAudio callback - runs in separate thread."""
        if status:
            logger.warning(f"Audio input status: {status}")

        # Notify all registered callbacks
        for callback in self._callbacks:
            try:
                callback(in_data)
            except Exception as e:
                logger.error(f"Audio callback error: {e}")

        return (None, pyaudio.paContinue if self._running else pyaudio.paComplete)

    def add_callback(self, callback: Callable[[bytes], None]) -> None:
        """Register a callback for audio data."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[bytes], None]) -> None:
        """Remove a registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def stream(self) -> AsyncIterator[bytes]:
        """Async generator yielding audio chunks."""
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        loop = asyncio.get_event_loop()

        def enqueue(data: bytes) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, data)
            except asyncio.QueueFull:
                pass  # Drop frames if consumer is slow

        self.add_callback(enqueue)
        try:
            while self._running:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=0.1)
                    yield data
                except asyncio.TimeoutError:
                    continue
        finally:
            self.remove_callback(enqueue)

    @staticmethod
    def bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
        """Convert 16-bit PCM bytes to float32 array [-1, 1]."""
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0

    @staticmethod
    def calculate_amplitude(audio_bytes: bytes) -> float:
        """Calculate RMS amplitude from audio bytes."""
        audio = AudioInput.bytes_to_float32(audio_bytes)
        return float(np.sqrt(np.mean(audio**2)))

    @property
    def is_running(self) -> bool:
        return self._running

    def __enter__(self) -> "AudioInput":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
