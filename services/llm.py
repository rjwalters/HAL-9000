"""
Groq LLM client with streaming.
"""

import asyncio
import logging
import re
from typing import AsyncIterator

from groq import AsyncGroq

import config

logger = logging.getLogger(__name__)


class GroqLLM:
    """Streaming LLM client using Groq."""

    def __init__(
        self,
        api_key: str = config.GROQ_API_KEY,
        model: str = config.GROQ_MODEL,
        max_tokens: int = config.GROQ_MAX_TOKENS,
        temperature: float = config.GROQ_TEMPERATURE,
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        self._client = AsyncGroq(api_key=self.api_key)

    async def generate(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
    ) -> str:
        """Generate a complete response (non-streaming)."""
        full_messages = []

        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})

        full_messages.extend(messages)

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=config.GROQ_TOP_P,
        )

        return response.choices[0].message.content

    async def generate_stream(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response, yielding tokens."""
        full_messages = []

        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})

        full_messages.extend(messages)

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=config.GROQ_TOP_P,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def generate_sentences(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Generate streaming response, yielding complete sentences.

        This is optimized for TTS - we yield complete sentences as soon
        as they're available to minimize latency.
        """
        buffer = ""
        sentence_end_pattern = re.compile(r"([.!?]+)\s*")

        async for token in self.generate_stream(messages, system_prompt):
            buffer += token

            # Check for sentence boundaries
            while True:
                match = sentence_end_pattern.search(buffer)
                if match:
                    # Found end of sentence
                    end_pos = match.end()
                    sentence = buffer[:end_pos].strip()
                    buffer = buffer[end_pos:]

                    if sentence:
                        logger.debug(f"Yielding sentence: {sentence}")
                        yield sentence
                else:
                    break

        # Yield any remaining text
        if buffer.strip():
            logger.debug(f"Yielding final: {buffer.strip()}")
            yield buffer.strip()

    async def generate_chunks(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        min_chunk_size: int = 20,
    ) -> AsyncIterator[str]:
        """
        Generate streaming response, yielding chunks suitable for TTS.

        Yields at sentence boundaries or when chunk reaches min size and
        hits a natural break point (comma, semicolon, etc).
        """
        buffer = ""
        # Sentence endings
        sentence_pattern = re.compile(r"([.!?]+)\s*")
        # Natural pause points (after reaching min size)
        pause_pattern = re.compile(r"([,;:\-\u2014])\s*")

        async for token in self.generate_stream(messages, system_prompt):
            buffer += token

            # Always yield at sentence boundaries
            while True:
                match = sentence_pattern.search(buffer)
                if match:
                    end_pos = match.end()
                    chunk = buffer[:end_pos].strip()
                    buffer = buffer[end_pos:]
                    if chunk:
                        yield chunk
                else:
                    break

            # If buffer is large enough, look for pause points
            if len(buffer) >= min_chunk_size:
                match = pause_pattern.search(buffer)
                if match:
                    end_pos = match.end()
                    chunk = buffer[:end_pos].strip()
                    buffer = buffer[end_pos:]
                    if chunk:
                        yield chunk

        # Yield remaining
        if buffer.strip():
            yield buffer.strip()
