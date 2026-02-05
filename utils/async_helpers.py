"""
Async utility functions and helpers.
"""

import asyncio
from asyncio import Queue
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, TypeVar

T = TypeVar("T")

# Shared thread pool for blocking operations
_executor = ThreadPoolExecutor(max_workers=4)


class AsyncQueue(Queue):
    """Enhanced async queue with additional utilities."""

    async def drain(self) -> list[Any]:
        """Drain all items currently in the queue."""
        items = []
        while not self.empty():
            try:
                items.append(self.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    def clear(self) -> None:
        """Clear all items from the queue."""
        while not self.empty():
            try:
                self.get_nowait()
            except asyncio.QueueEmpty:
                break


async def run_in_thread(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking function in a thread pool."""
    loop = asyncio.get_event_loop()
    if kwargs:
        func = partial(func, **kwargs)
    return await loop.run_in_executor(_executor, func, *args)


class StreamBuffer:
    """Buffer for managing streaming data with backpressure."""

    def __init__(self, max_size: int = 100):
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=max_size)
        self._closed = False

    async def write(self, data: bytes) -> None:
        """Write data to buffer."""
        if not self._closed:
            await self._queue.put(data)

    async def read(self) -> bytes | None:
        """Read data from buffer. Returns None when closed."""
        return await self._queue.get()

    def close(self) -> None:
        """Signal end of stream."""
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    @property
    def closed(self) -> bool:
        return self._closed


def chunk_text_by_sentence(text: str) -> list[str]:
    """Split text into sentences for streaming TTS."""
    import re

    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]
