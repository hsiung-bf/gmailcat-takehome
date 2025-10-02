"""
Generic OpenAI LLM client (reusable across projects).

Handles API calls, retry logic, and error handling.
"""

import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential

from openai import OpenAI


@dataclass
class LLMCompletionResult:
    """Result of a single llm chat completion."""
    success: bool
    content: Optional[str] = None
    error: Optional[str] = None


class LLMClient:
    """Generic OpenAI API client with retry logic."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def chat_completion(
        self, 
        messages: Union[str, List[Dict[str, str]]], 
        max_tokens: int = 500, 
        temperature: float = 0.1,
        model: Optional[str] = None,
        json_mode: bool = False
    ) -> str:
        """
        Generate a single chat completion with retry logic.
        
        Args:
            messages: Either a string (user prompt) or list of message dicts
                     [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            model: Model to use
            json_mode: If True, force JSON response format
        
        Raises:
            Exception: If API call fails after 3 retries
        """
        # Handle backward compatibility: string -> user message
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
    
    def chat_completion_batch(
        self,
        messages_list: List[Union[str, List[Dict[str, str]]]],
        max_tokens: int = 500,
        temperature: float = 0.1,
        model: Optional[str] = None,
        max_workers: int = 5,
        json_mode: bool = False
    ) -> List[LLMCompletionResult]:
        """
        Generate completions for multiple message sets in parallel.
        
        Args:
            messages_list: List of messages (each can be a string or list of message dicts)
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature
            model: Override default model
            max_workers: Number of parallel threads
            json_mode: If True, force JSON response format
            
        Returns:
            List of LLMCompletionResult objects (in same order as messages_list)
        """
        results = [None] * len(messages_list)
        
        def worker(i, messages):
            """Process a single message set and return its index and result."""
            try:
                content = self.chat_completion(messages, max_tokens, temperature, model, json_mode)
                return i, LLMCompletionResult(success=True, content=content)
            except Exception as e:
                return i, LLMCompletionResult(success=False, error=str(e))
        
        with ThreadPoolExecutor(max_workers=min(len(messages_list), max_workers)) as executor:
            futures = [executor.submit(worker, i, messages) for i, messages in enumerate(messages_list)]
            
            for future in as_completed(futures):
                i, result = future.result()
                results[i] = result
        
        return results