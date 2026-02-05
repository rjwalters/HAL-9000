"""HAL-9000 personality and conversation management."""

from .personality import get_system_prompt, get_greeting, get_error_response
from .conversation import ConversationManager
from .transcript_logger import TranscriptLogger

__all__ = ["get_system_prompt", "get_greeting", "get_error_response", "ConversationManager", "TranscriptLogger"]
