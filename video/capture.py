"""
Video capture using OpenCV.
"""

import asyncio
import base64
import logging
import threading
import time
from io import BytesIO

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class VideoCapture:
    """Webcam capture with frame queue management."""

    def __init__(
        self,
        device_id: int = config.VIDEO_DEVICE_ID,
        width: int = config.VIDEO_FRAME_WIDTH,
        height: int = config.VIDEO_FRAME_HEIGHT,
        fps: int = config.VIDEO_FPS,
        jpeg_quality: int = config.VISION_JPEG_QUALITY,
    ):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality

        self._capture: cv2.VideoCapture | None = None
        self._running = False
        self._thread: threading.Thread | None = None

        # Latest frame storage
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_time: float = 0
        self._frame_lock = threading.Lock()

    def start(self) -> bool:
        """Start video capture in background thread. Returns True if successful."""
        if self._running:
            return True

        self._capture = cv2.VideoCapture(self.device_id)

        if not self._capture.isOpened():
            logger.warning(f"Failed to open camera device {self.device_id} - vision disabled")
            self._capture = None
            return False

        # Configure capture
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        logger.info(
            f"Video capture started: device={self.device_id}, "
            f"{self.width}x{self.height}@{self.fps}fps"
        )
        return True

    def stop(self) -> None:
        """Stop video capture."""
        self._running = False

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._capture:
            self._capture.release()
            self._capture = None

        logger.info("Video capture stopped")

    def _capture_loop(self) -> None:
        """Background thread capturing frames."""
        while self._running and self._capture:
            ret, frame = self._capture.read()
            if ret:
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_frame_time = time.time()
            else:
                logger.warning("Failed to capture frame")
                time.sleep(0.01)

    def get_frame(self) -> np.ndarray | None:
        """Get the latest captured frame."""
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def get_frame_age(self) -> float:
        """Get age of latest frame in seconds."""
        with self._frame_lock:
            if self._latest_frame_time > 0:
                return time.time() - self._latest_frame_time
            return float("inf")

    def get_frame_jpeg(self) -> bytes | None:
        """Get latest frame as JPEG bytes."""
        frame = self.get_frame()
        if frame is None:
            return None

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if success:
            return buffer.tobytes()
        return None

    def get_frame_base64(self) -> str | None:
        """Get latest frame as base64-encoded JPEG."""
        jpeg_bytes = self.get_frame_jpeg()
        if jpeg_bytes:
            return base64.b64encode(jpeg_bytes).decode("utf-8")
        return None

    async def get_frame_async(self) -> np.ndarray | None:
        """Async wrapper for get_frame."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_frame)

    async def get_frame_base64_async(self) -> str | None:
        """Async wrapper for get_frame_base64."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_frame_base64)

    @property
    def is_running(self) -> bool:
        return self._running

    def __enter__(self) -> "VideoCapture":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
