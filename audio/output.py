"""
Audio output playback using PyAudio.
"""

import asyncio
import logging
import threading
from collections import deque

import numpy as np
import pyaudio

import config

logger = logging.getLogger(__name__)


class AudioOutput:
    """Speaker playback with streaming buffer."""

    def __init__(
        self,
        sample_rate: int = config.PLAYBACK_SAMPLE_RATE,
        channels: int = config.PLAYBACK_CHANNELS,
        chunk_size: int = config.PLAYBACK_CHUNK_SIZE,
        device_index: int | None = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index

        self._pyaudio: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None
        self._running = False

        # Thread-safe buffer for audio data
        self._buffer: deque[bytes] = deque(maxlen=500)
        self._buffer_lock = threading.Lock()

        # Current amplitude for visualization
        self._current_amplitude = 0.0
        self._amplitude_lock = threading.Lock()

    def start(self) -> None:
        """Start audio playback."""
        if self._running:
            return

        self._pyaudio = pyaudio.PyAudio()

        # Log available devices
        logger.info("Available audio output devices:")
        for i in range(self._pyaudio.get_device_count()):
            info = self._pyaudio.get_device_info_by_index(i)
            if info["maxOutputChannels"] > 0:
                logger.info(f"  [{i}] {info['name']}")

        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            output_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback,
        )

        self._running = True
        logger.info(
            f"Audio output started: {self.sample_rate}Hz, "
            f"{self.channels}ch, chunk={self.chunk_size}"
        )

    def stop(self) -> None:
        """Stop audio playback."""
        self._running = False

        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None

        logger.info("Audio output stopped")

    def _audio_callback(
        self,
        in_data: bytes,
        frame_count: int,
        time_info: dict,
        status: int,
    ) -> tuple[bytes, int]:
        """PyAudio callback - runs in separate thread."""
        if status:
            logger.warning(f"Audio output status: {status}")

        # Calculate bytes needed
        bytes_needed = frame_count * self.channels * 2  # 16-bit = 2 bytes

        # Collect data from buffer
        data = b""
        with self._buffer_lock:
            while len(data) < bytes_needed and self._buffer:
                chunk = self._buffer.popleft()
                data += chunk

        # Pad with silence if not enough data
        if len(data) < bytes_needed:
            data += b"\x00" * (bytes_needed - len(data))
        elif len(data) > bytes_needed:
            # Put excess back
            with self._buffer_lock:
                self._buffer.appendleft(data[bytes_needed:])
            data = data[:bytes_needed]

        # Calculate amplitude for visualization
        self._update_amplitude(data)

        return (data, pyaudio.paContinue if self._running else pyaudio.paComplete)

    def _update_amplitude(self, data: bytes) -> None:
        """Update current amplitude from audio data."""
        if len(data) < 2:
            return

        audio_int16 = np.frombuffer(data, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        amplitude = float(np.sqrt(np.mean(audio_float**2)))

        with self._amplitude_lock:
            # Smooth amplitude changes
            self._current_amplitude = 0.7 * amplitude + 0.3 * self._current_amplitude

    def write(self, data: bytes) -> None:
        """Write audio data to playback buffer."""
        with self._buffer_lock:
            self._buffer.append(data)

    async def write_async(self, data: bytes) -> None:
        """Async write to playback buffer."""
        self.write(data)

    def clear_buffer(self) -> None:
        """Clear the playback buffer."""
        with self._buffer_lock:
            self._buffer.clear()
        with self._amplitude_lock:
            self._current_amplitude = 0.0

    def get_amplitude(self) -> float:
        """Get current audio amplitude (0-1 range)."""
        with self._amplitude_lock:
            return min(1.0, self._current_amplitude * 3)  # Scale up for visibility

    def get_buffer_level(self) -> int:
        """Get current buffer level (number of chunks)."""
        with self._buffer_lock:
            return len(self._buffer)

    def is_playing(self) -> bool:
        """Check if there's audio in the buffer."""
        with self._buffer_lock:
            return len(self._buffer) > 0

    @property
    def is_running(self) -> bool:
        return self._running

    def __enter__(self) -> "AudioOutput":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
