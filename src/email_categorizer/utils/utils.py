import json
from typing import Any, Optional
from ..types import Message


def truncate_text(text: str, max_chars: int = 2000) -> str:
    """
    Truncate text to specified character limit, preserving word boundaries.
    
    Args:
        text: Text to truncate
        max_chars: Maximum number of characters
        
    Returns:
        Truncated text with "..." if truncated
    """
    if not text or len(text) <= max_chars:
        return text
    
    # Find the last space before the limit to avoid cutting words
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    
    if last_space > max_chars * 0.8:  # Only use word boundary if it's not too far back
        truncated = truncated[:last_space]
    
    return truncated + "..."


def extract_json_from_response(response: str) -> Optional[Any]:
    """
    Extract JSON object from LLM response text.
    
    Args:
        response: Raw response text from LLM
        
    Returns:
        Parsed JSON object
    
    Raises: error if the response if a valid JSON obejct cannot be extracted
    """
    # First try parsing the entire response as JSON
    try:
        return json.loads(response)
    except:
        pass
    
    # Try to extract JSON from within the response
    json_start = response.find('{')
    json_end = response.rfind('}') + 1
    
    if json_start >= 0 and json_end > json_start:
        try:
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        except:
            pass
    
    raise json.JSONDecodeError("No valid JSON object found in response", response, 0)


def format_email_preview(message: Message) -> str:
    """Format an email message as a text preview."""
    lines = [
        f"From: {message.sender}",
        f"Subject: {message.subject}",
        f"Date: {message.date}",
        f"Preview: {message.preview_text}"
    ]
    return "\n".join(lines)


def get_compact_message_representation(message: Message, max_body_length: int = 1000) -> str:
    """
    Get structured text representation of a message for embedding/similarity.
    
    Args:
        message: Message object
        max_body_length: Maximum length for body text (default: 1000)
        
    Returns:
        Structured text representation with subject, sender, and truncated body
    """
    # Truncate body text
    body_text = message.body_text or ""
    if len(body_text) > max_body_length:
        body_text = truncate_text(body_text, max_body_length)
    
    # Create structured format
    lines = [
        f"Subject: {message.subject or ''}",
        f"From: {message.sender or ''}",
        f"Body: {body_text}"
    ]
    
    return "\n".join(lines)

