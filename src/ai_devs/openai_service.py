"""OpenRouter API service wrapper with structured output support."""

import json
from typing import Type, TypeVar, Optional
from openai import OpenAI
from pydantic import BaseModel

from .config import get_open_router_api_key, get_open_router_base_url

T = TypeVar("T", bound=BaseModel)


class OpenRouterService:
    """Wrapper around OpenRouter API (OpenAI-compatible)."""

    def __init__(self, model: str = "gpt-4.1-mini", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or get_open_router_api_key(),
            base_url=base_url or get_open_router_base_url(),
        )

    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs,
    ) -> str:
        """Send a chat completion request and return the text response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Override the default model
            temperature: Sampling temperature
            
        Returns:
            The assistant's response text
        """
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content

    def structured_output(
        self,
        messages: list[dict],
        response_schema: Type[T],
        model: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs,
    ) -> T:
        """Send a chat completion with structured output (JSON Schema).
        
        Uses OpenAI's structured output feature to guarantee the response
        matches the provided Pydantic model schema.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            response_schema: A Pydantic model class defining the expected response
            model: Override the default model
            temperature: Sampling temperature
            
        Returns:
            An instance of the response_schema Pydantic model
        """
        response = self.client.beta.chat.completions.parse(
            model=model or self.model,
            messages=messages,
            response_format=response_schema,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.parsed

    def simple_query(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Quick single-turn query to the model.
        
        Args:
            prompt: The user's message
            system: System prompt
            model: Override the default model
            temperature: Sampling temperature
            
        Returns:
            The assistant's response text
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self.chat(messages, model=model, temperature=temperature)
