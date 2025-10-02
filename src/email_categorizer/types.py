from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Message:
    msg_id: str
    sender: str
    recipients: List[str]
    date: str
    subject: str
    preview_text: str # short preview text is the beginning of the body
    body_text: str # full cleaned text

@dataclass
class Category:
    name: str # provided by user
    description: str # provided by user
    slug: str # short, filesystem/URL-safe identifier you derive from the category name (lowercase, hyphens, no spaces)
    keywords: List[str] # LLM-generated keywords (required)
    
@dataclass
class ClassificationResult:
    msg_id: str
    category_slug: str
    is_in_category: bool
    explanation: str