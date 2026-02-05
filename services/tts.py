"""
ElevenLabs Text-to-Speech streaming client.
"""

import asyncio
import logging
from typing import AsyncIterator

from elevenlabs import AsyncElevenLabs, VoiceSettings

import config

logger = logging.getLogger(__name__)


class ElevenLabsTTS:
    """Streaming TTS using ElevenLabs."""

    def __init__(
        self,
        api_key: str = config.ELEVENLABS_API_KEY,
        voice_id: str = config.ELEVENLABS_VOICE_ID,
        model: str = config.ELEVENLABS_MODEL,
    ):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model

        self._client = AsyncElevenLabs(api_key=self.api_key)

        # Voice settings for HAL's calm, measured tone
        self._voice_settings = VoiceSettings(
            stability=config.ELEVENLABS_STABILITY,
            similarity_boost=config.ELEVENLABS_SIMILARITY_BOOST,
            style=config.ELEVENLABS_STYLE,
            use_speaker_boost=True,
        )

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio bytes (non-streaming)."""
        audio_data = b""

        async for chunk in self._client.text_to_speech.convert(
            voice_id=self.voice_id,
            text=text,
            model_id=self.model,
            voice_settings=self._voice_settings,
            output_format="pcm_24000",  # 24kHz 16-bit PCM
        ):
            audio_data += chunk

        return audio_data

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text to streaming audio chunks."""
        async for chunk in self._client.text_to_speech.convert(
            voice_id=self.voice_id,
            text=text,
            model_id=self.model,
            voice_settings=self._voice_settings,
            output_format="pcm_24000",
            optimize_streaming_latency=config.ELEVENLABS_OPTIMIZE_LATENCY,
        ):
            yield chunk

    async def synthesize_sentences(
        self,
        sentence_stream: AsyncIterator[str],
    ) -> AsyncIterator[bytes]:
        """
        Synthesize a stream of sentences to audio.

        This processes sentences as they arrive, yielding audio chunks
        with minimal latency.
        """
        async for sentence in sentence_stream:
            if not sentence.strip():
                continue

            logger.debug(f"Synthesizing: {sentence}")

            async for chunk in self.synthesize_stream(sentence):
                yield chunk

    async def synthesize_to_queue(
        self,
        text: str,
        audio_queue: asyncio.Queue,
    ) -> None:
        """Synthesize text and put audio chunks in queue."""
        async for chunk in self.synthesize_stream(text):
            await audio_queue.put(chunk)

        # Signal end of audio
        await audio_queue.put(None)

    async def get_voices(self) -> list[dict]:
        """Get available voices."""
        response = await self._client.voices.get_all()
        return [
            {"id": v.voice_id, "name": v.name, "category": v.category}
            for v in response.voices
        ]
