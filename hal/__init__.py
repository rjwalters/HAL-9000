"""HAL-9000 personality and conversation management."""

from .personality import get_system_prompt, get_greeting, get_error_response
from .conversation import ConversationManager

__all__ = ["get_system_prompt", "get_greeting", "get_error_response", "ConversationManager"]
