"""
Conversation history management for HAL-9000.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "user", "assistant", or "vision"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    display_only: bool = False

    def to_dict(self) -> dict:
        """Convert to LLM API format."""
        if self.role == "vision":
            return {"role": "user", "content": f"[Background visual context — do not describe this aloud, use it to inform your responses] {self.content}"}
        return {"role": self.role, "content": self.content}


class ConversationManager:
    """Manages conversation history with HAL-9000."""

    def __init__(
        self,
        max_messages: int = 20,
        max_tokens_estimate: int = 2000,
    ):
        """
        Initialize conversation manager.

        Args:
            max_messages: Maximum number of messages to retain
            max_tokens_estimate: Rough token limit for context
        """
        self.max_messages = max_messages
        self.max_tokens_estimate = max_tokens_estimate
        self._messages: List[Message] = []

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        message = Message(role="user", content=content)
        self._messages.append(message)
        self._prune_history()
        logger.debug(f"Added user message: {content[:50]}...")

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant (HAL) message to the conversation."""
        message = Message(role="assistant", content=content)
        self._messages.append(message)
        self._prune_history()
        logger.debug(f"Added assistant message: {content[:50]}...")

    def add_vision_message(self, content: str) -> None:
        """Add a vision description to conversation history."""
        message = Message(role="vision", content=content)
        self._messages.append(message)
        self._prune_history()
        logger.debug(f"Added vision message: {content[:50]}...")

    def get_messages(self) -> List[dict]:
        """Get conversation history in LLM API format."""
        return [msg.to_dict() for msg in self._messages]

    def get_messages_with_timestamps(self) -> list[dict]:
        """Get conversation history with timestamps for display."""
        return [
            {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp.strftime("%H:%M:%S")}
            for msg in self._messages
        ]

    def get_last_user_message(self) -> Optional[str]:
        """Get the most recent user message."""
        for msg in reversed(self._messages):
            if msg.role == "user":
                return msg.content
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the most recent assistant message."""
        for msg in reversed(self._messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def _prune_history(self) -> None:
        """Prune history to stay within limits."""
        # Remove oldest messages if over limit
        while len(self._messages) > self.max_messages:
            removed = self._messages.pop(0)
            logger.debug(f"Pruned old message: {removed.content[:30]}...")

        # Rough token estimation (4 chars ≈ 1 token)
        total_chars = sum(len(msg.content) for msg in self._messages)
        estimated_tokens = total_chars / 4

        while estimated_tokens > self.max_tokens_estimate and len(self._messages) > 2:
            removed = self._messages.pop(0)
            total_chars -= len(removed.content)
            estimated_tokens = total_chars / 4
            logger.debug(f"Pruned for token limit: {removed.content[:30]}...")

    def clear(self) -> None:
        """Clear all conversation history."""
        self._messages.clear()
        logger.info("Conversation history cleared")

    def get_summary(self) -> str:
        """Get a brief summary of the conversation state."""
        if not self._messages:
            return "No conversation history"

        return (
            f"{len(self._messages)} messages, "
            f"last: {self._messages[-1].role} at "
            f"{self._messages[-1].timestamp.strftime('%H:%M:%S')}"
        )

    @property
    def message_count(self) -> int:
        """Get the number of messages in history."""
        return len(self._messages)

    @property
    def is_empty(self) -> bool:
        """Check if conversation history is empty."""
        return len(self._messages) == 0
