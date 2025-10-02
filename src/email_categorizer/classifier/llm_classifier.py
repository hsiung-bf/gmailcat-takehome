"""
LLM-based email classification using OpenAI API.

Supports batch processing for efficiency.
"""

import json
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..llm_client.llm_client import LLMClient, LLMCompletionResult
from ..types import Message, Category, ClassificationResult
from ..utils.utils import extract_json_from_response

MAX_WORKERS = 3
EMAILS_PER_BATCH = 3 # number of emails to put in a single llm request
PROMPT_TEMPLATES_PATH = Path(__file__).parent.parent / "llm_client" / "prompts.yaml"


class LLMClassifier:
    """Classify emails using an LLM."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the LLM classifier.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: OpenAI model to use (default: gpt-4o-mini)
        """
        self.llm_client = LLMClient(api_key=api_key)
        self.model = model
        self.batch_size = EMAILS_PER_BATCH
        self.max_workers = MAX_WORKERS
        
        try:
            with open(PROMPT_TEMPLATES_PATH, 'r') as f:
                templates = yaml.safe_load(f)
                self.email_template = templates["email_template"]
                self.batch_prompts = templates["batch_classification_prompts"]
        except Exception as e:
            raise ValueError(f"Error loading prompt templates: {e}")    

    def classify_emails(self, emails: List[Message], category: Category) -> Dict[str, Optional[ClassificationResult]]:
        """
        Classify a list of emails using batch processing and parallelization.
        
        Args:
            emails: List of emails to classify
            category: Category to classify against
            
        Returns:
            Dictionary mapping email msg_id to classification result (or None if there was an error classifying the email)
        """
        if not emails:
            return {}
        
        batches: List[List[Message]] = self._split_into_batches(emails)
        batch_messages: List[List[Dict[str, str]]] = [self._create_batch_messages(batch, category) for batch in batches]
        
        completion_results: List[LLMCompletionResult] = self.llm_client.chat_completion_batch(
            messages_list=batch_messages,
            max_tokens=500,
            temperature=0.1,
            model=self.model,
            max_workers=self.max_workers,
            json_mode=True
        )

        all_results = {}
        for completion_result, email_batch in zip(completion_results, batches):
            batch_results = self._parse_batch_response(completion_result, email_batch, category)
            all_results.update(batch_results)
        return all_results
    
    def _split_into_batches(self, emails: List[Message]) -> List[List[Message]]:
        """Split emails into batches."""
        # TODO: dynamically create batches based on the lengths of the emails
        batches = []
        for i in range(0, len(emails), self.batch_size):
            batch = emails[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    def _create_batch_messages(self, emails: List[Message], category: Category) -> List[Dict[str, str]]:
        """Create system and user messages for batch classification.""" 
        system_prompt = self.batch_prompts["system_prompt"]
        user_prompt = self.batch_prompts["user_prompt"].format(
            category_name=category.name,
            category_description=category.description,
            formatted_emails=self._format_emails(emails)
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _format_emails(self, emails: List[Message]) -> str:
        """Format the emails as JSON."""
        formatted_emails = []
        for i, email in enumerate(emails, 1):
            body = email.body_text[:500]
            email_dict = {
                "email_number": i,
                "sender": email.sender,
                "recipients": email.recipients,
                "date": email.date,
                "subject": email.subject,
                "body": body
            }
            formatted_emails.append(json.dumps(email_dict, indent=2))
        return ",\n".join(formatted_emails)
    
    def _parse_batch_response(self, completion_result: LLMCompletionResult, emails: List[Message], category: Category) -> Dict[str, Optional[ClassificationResult]]:
        """Parse the JSON batch response from the LLM for a single batch of emails."""
        results = {}

        # LLM failure - return None for all emails
        if not completion_result.success:
            for email in emails:
                results[email.msg_id] = None
            return results
    
        try: # Parse JSON response
            response_json = extract_json_from_response(completion_result.content)
            classifications = response_json["classifications"]
            if len(classifications) != len(emails):
                raise ValueError(
                    f"LLM returned {len(classifications)} classifications but expected {len(emails)}"
                )
            
            # Match classifications to emails in order
            for email, classification in zip(emails, classifications):
                results[email.msg_id] = ClassificationResult(
                    msg_id=email.msg_id,
                    category_slug=category.slug,
                    is_in_category=classification["is_in_category"],
                    explanation=classification["explanation"]
                )
        except Exception as e: # JSON parsing failed. Return None for all emails
            print(f"Failed to parse batch response: {e}")
            for email in emails:
                results[email.msg_id] = None
        
        return results