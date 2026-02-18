"""
Deepgram Speech-to-Text streaming client.
Uses websockets directly for real-time transcription.
"""

import asyncio
import json
import logging
from typing import Callable

import websockets

import config

logger = logging.getLogger(__name__)

DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"


class DeepgramSTT:
    """Streaming speech-to-text using Deepgram WebSocket API."""

    def __init__(
        self,
        api_key: str = config.DEEPGRAM_API_KEY,
        model: str = config.DEEPGRAM_MODEL,
        language: str = config.DEEPGRAM_LANGUAGE,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
    ):
        self.api_key = api_key
        self.model = model
        self.language = language
        self.sample_rate = sample_rate

        self._websocket = None
        self._receive_task = None
        self._transcript_buffer: list[str] = []
        self._final_transcript: str = ""

        # Utterance end event — set when Deepgram detects end of speech
        self._utterance_end_event: asyncio.Event | None = None

        # Callbacks
        self._on_transcript: Callable[[str, bool], None] | None = None

    def _build_url(self) -> str:
        """Build WebSocket URL with query parameters."""
        params = [
            f"model={self.model}",
            f"language={self.language}",
            f"encoding=linear16",
            f"sample_rate={self.sample_rate}",
            f"channels=1",
            f"smart_format={str(config.DEEPGRAM_SMART_FORMAT).lower()}",
            f"interim_results={str(config.DEEPGRAM_INTERIM_RESULTS).lower()}",
            f"endpointing=300",
            f"utterance_end_ms=750",
        ]
        return f"{DEEPGRAM_WS_URL}?{'&'.join(params)}"

    async def connect(self) -> None:
        """Establish WebSocket connection to Deepgram."""
        if self._websocket:
            return

        url = self._build_url()
        headers = {"Authorization": f"Token {self.api_key}"}

        try:
            self._websocket = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            )
            logger.info("Deepgram WebSocket connected")

            # Start receiving messages
            self._receive_task = asyncio.create_task(self._receive_loop())

        except Exception as e:
            logger.error(f"Failed to connect to Deepgram: {e}")
            raise

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._websocket:
            await self._websocket.close()
            self._websocket = None
            logger.info("Deepgram WebSocket disconnected")

    async def _receive_loop(self) -> None:
        """Background task to receive transcripts."""
        try:
            async for message in self._websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from Deepgram: {message}")
        except websockets.exceptions.ConnectionClosed:
            logger.debug("Deepgram WebSocket closed")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Deepgram receive error: {e}")

    async def _handle_message(self, data: dict) -> None:
        """Handle incoming Deepgram message."""
        msg_type = data.get("type")

        if msg_type == "Results":
            # Extract transcript
            channel = data.get("channel", {})
            alternatives = channel.get("alternatives", [])

            if alternatives:
                transcript = alternatives[0].get("transcript", "")
                is_final = data.get("is_final", False)

                if transcript:
                    if is_final:
                        self._transcript_buffer.append(transcript)
                        self._final_transcript = " ".join(self._transcript_buffer)
                        logger.debug(f"Final: {transcript}")
                    else:
                        logger.debug(f"Interim: {transcript}")

                    if self._on_transcript:
                        self._on_transcript(transcript, is_final)

        elif msg_type == "Metadata":
            logger.debug(f"Deepgram metadata: {data}")

        elif msg_type == "UtteranceEnd":
            logger.info("Deepgram utterance end detected")
            if self._utterance_end_event:
                self._utterance_end_event.set()

        elif msg_type == "SpeechStarted":
            logger.debug("Speech started")

    async def send_audio(self, audio_bytes: bytes) -> None:
        """Send audio chunk to Deepgram."""
        if self._websocket:
            try:
                await self._websocket.send(audio_bytes)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Cannot send audio - connection closed")

    def on_transcript(self, callback: Callable[[str, bool], None]) -> None:
        """Register callback for transcript events."""
        self._on_transcript = callback

    def get_final_transcript(self) -> str:
        """Get the accumulated final transcript."""
        return self._final_transcript

    def clear_transcript(self) -> None:
        """Clear the transcript buffer and reset utterance end event."""
        self._transcript_buffer.clear()
        self._final_transcript = ""
        self._utterance_end_event = asyncio.Event()

    @property
    def utterance_end_event(self) -> asyncio.Event:
        """Event that is set when Deepgram detects end of utterance."""
        if self._utterance_end_event is None:
            self._utterance_end_event = asyncio.Event()
        return self._utterance_end_event

    async def transcribe_stream(
        self,
        audio_stream,
        end_signal: asyncio.Event,
    ) -> str:
        """
        Transcribe an audio stream until end signal is set.

        Args:
            audio_stream: Async iterator yielding audio bytes
            end_signal: Event to signal end of speech

        Returns:
            Final transcript
        """
        await self.connect()
        self.clear_transcript()

        try:
            async for audio_chunk in audio_stream:
                await self.send_audio(audio_chunk)

                if end_signal.is_set():
                    break

                await asyncio.sleep(0)

            # Wait for final transcript
            await asyncio.sleep(0.3)
            return self.get_final_transcript()

        finally:
            await self.disconnect()
