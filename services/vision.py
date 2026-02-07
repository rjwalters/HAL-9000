"""
Claude Vision API client for scene description.
"""

import asyncio
import base64
import logging
import time
from typing import Optional

import anthropic
import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class ClaudeVision:
    """Vision API client with caching for scene descriptions."""

    def __init__(
        self,
        api_key: str = config.ANTHROPIC_API_KEY,
        model: str = config.VISION_MODEL,
        max_tokens: int = config.VISION_MAX_TOKENS,
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens

        self._client = anthropic.AsyncAnthropic(api_key=self.api_key)

        # Cached scene description
        self._cached_description: str = "No visual information available."
        self._cache_time: float = 0
        self._cache_lock = asyncio.Lock()

        # Frame differencing
        self._last_sent_frame: np.ndarray | None = None
        self._diff_threshold: float = config.VISION_DIFF_THRESHOLD

    async def describe_image(
        self,
        image_base64: str,
        media_type: str = "image/jpeg",
    ) -> str:
        """Get a description of an image."""
        message = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "In 10 words or fewer, note who is present and "
                                "what they are doing. No detail about the setting."
                            ),
                        },
                    ],
                }
            ],
        )

        return message.content[0].text

    async def update_cache(self, image_base64: str) -> str:
        """Update the cached scene description."""
        async with self._cache_lock:
            try:
                description = await self.describe_image(image_base64)
                self._cached_description = description
                self._cache_time = time.time()
                logger.debug(f"Vision cache updated: {description[:50]}...")
                return description
            except Exception as e:
                logger.error(f"Vision API error: {e}")
                return self._cached_description

    def get_cached_description(self) -> str:
        """Get the most recent scene description."""
        return self._cached_description

    def get_cache_age(self) -> float:
        """Get age of cached description in seconds."""
        if self._cache_time > 0:
            return time.time() - self._cache_time
        return float("inf")

    def _frame_changed(self, frame_base64: str) -> bool:
        """Check if the frame differs enough from the last sent frame."""
        jpeg_bytes = base64.b64decode(frame_base64)
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

        if self._last_sent_frame is None:
            self._last_sent_frame = frame
            return True

        diff = cv2.absdiff(frame, self._last_sent_frame)
        mean_diff = float(np.mean(diff))
        logger.debug(f"Frame diff: {mean_diff:.1f} (threshold={self._diff_threshold})")

        if mean_diff >= self._diff_threshold:
            self._last_sent_frame = frame
            return True

        return False

    async def run_vision_loop(
        self,
        get_frame_func,
        interval: float = config.VISION_CAPTURE_INTERVAL,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """
        Background task that continuously updates vision cache.

        Args:
            get_frame_func: Async function that returns base64-encoded frame
            interval: Seconds between vision API calls
            stop_event: Event to signal loop should stop
        """
        logger.info(f"Starting vision loop (interval={interval}s, diff_threshold={self._diff_threshold})")

        while not (stop_event and stop_event.is_set()):
            try:
                # Get latest frame
                frame_base64 = await get_frame_func()

                if frame_base64:
                    if self._frame_changed(frame_base64):
                        await self.update_cache(frame_base64)
                    else:
                        logger.debug("Scene unchanged, skipping vision API call")
                else:
                    logger.warning("No frame available for vision")

            except Exception as e:
                logger.error(f"Vision loop error: {e}")

            # Wait for next interval
            try:
                if stop_event:
                    await asyncio.wait_for(
                        stop_event.wait(),
                        timeout=interval,
                    )
                    break
                else:
                    await asyncio.sleep(interval)
            except asyncio.TimeoutError:
                continue

        logger.info("Vision loop stopped")
