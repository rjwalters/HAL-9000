"""
Transcript logging for troubleshooting.
Saves all conversations with timestamps.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class TranscriptLogger:
    """Logs all conversations to a JSON Lines file."""

    def __init__(self, log_dir: str = "transcripts"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create new log file for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"session_{timestamp}.jsonl"
        self.session_start = datetime.now()

        logger.info(f"Transcript logging to: {self.log_file}")

    def log_event(self, event_type: str, data: dict) -> None:
        """Log an event with timestamp."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": (datetime.now() - self.session_start).total_seconds(),
            "event": event_type,
            **data,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_user_speech(self, transcript: str, duration_ms: int | None = None) -> None:
        """Log user speech transcript."""
        self.log_event("user_speech", {
            "transcript": transcript,
            "duration_ms": duration_ms,
        })
        logger.info(f"[USER] {transcript}")

    def log_hal_response(self, response: str, latency_ms: int | None = None) -> None:
        """Log HAL's response."""
        self.log_event("hal_response", {
            "response": response,
            "latency_ms": latency_ms,
        })
        logger.info(f"[HAL] {response}")

    def log_vision(self, description: str) -> None:
        """Log vision description."""
        self.log_event("vision", {
            "description": description,
        })

    def log_state_change(self, from_state: str, to_state: str) -> None:
        """Log state transitions."""
        self.log_event("state_change", {
            "from": from_state,
            "to": to_state,
        })

    def log_error(self, error_type: str, message: str) -> None:
        """Log errors."""
        self.log_event("error", {
            "error_type": error_type,
            "message": message,
        })
        logger.error(f"[ERROR] {error_type}: {message}")

    def log_latency(self, component: str, latency_ms: float) -> None:
        """Log component latency for performance analysis."""
        self.log_event("latency", {
            "component": component,
            "latency_ms": round(latency_ms, 2),
        })
