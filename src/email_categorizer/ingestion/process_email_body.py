import base64
import re
from typing import Optional

from bs4 import BeautifulSoup
from ..utils.utils import truncate_text

def process_email_body(body_b64: str, max_chars: int = 2000) -> tuple[str, str]:
    html = decode_body_base64(body_b64)
    text = html_to_text(html)
    truncated_text = truncate_text(text, max_chars)
    return text, truncated_text 


def decode_body_base64(body_b64: str) -> str:
    """
    Decode base64-encoded email body to HTML string.
    
    Args:
        body_b64: Base64-encoded HTML content
        
    Returns:
        Decoded HTML string
        
    Raises:
        ValueError: If base64 decoding fails
    """
    try:
        decoded_bytes = base64.b64decode(body_b64)
        return decoded_bytes.decode('utf-8', errors='replace')
    except Exception as e:
        raise ValueError(f"Failed to decode base64 body: {e}")


def html_to_text(html: str) -> str:
    """
    Convert HTML to clean plain text.
    
    - Removes HTML tags while preserving structure
    - Handles HTML entities automatically
    - Converts links to readable format
    - Preserves line breaks and paragraphs
    
    Args:
        html: HTML string
        
    Returns:
        Clean plain text
    """
    if not html:
        return ""
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Convert links to readable format
    for link in soup.find_all('a'):
        href = link.get('href', '')
        text = link.get_text(strip=True)
        if href and text:
            link.replace_with(f"{text} ({href})")
        elif text:
            link.replace_with(text)
        else:
            link.decompose()
    
    # Replace images with placeholders
    for img in soup.find_all('img'):
        alt_text = img.get('alt', '').strip()
        if alt_text and len(alt_text) > 3:  # Keep meaningful alt text
            img.replace_with(f"[Image: {alt_text}]")
        else:
            img.replace_with("[Image]")
    
    # Convert block elements to line breaks
    for tag in soup.find_all(['p', 'div', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        if tag.name == 'br':
            tag.replace_with('\n')
        else:
            tag.insert_after('\n\n')
    
    # Convert list items
    for li in soup.find_all('li'):
        li.insert_before('• ')
        li.insert_after('\n')
    
    # Get text content
    text = soup.get_text()
    
    # Clean up whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
    text = text.strip()
    
    return text