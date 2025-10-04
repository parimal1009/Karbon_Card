"""LLM Client for AI Agent with Groq integration."""

import os
from typing import Optional, Iterator
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from loguru import logger


class LLMClient:
    """Wrapper for LLM interactions with error handling and retries."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 8000
    ):
        """
        Initialize LLM client.
        
        Args:
            api_key: Groq API key (defaults to env var)
            model_name: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        logger.info(f"Initialized LLM client with model: {self.model_name}")
    
    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        """
        Invoke LLM with system and user prompts.
        
        Args:
            system_prompt: System instructions
            user_prompt: User query
            
        Returns:
            LLM response text
        """
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise
    
    def stream_invoke(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """
        Stream LLM response for real-time feedback.
        
        Args:
            system_prompt: System instructions
            user_prompt: User query
            
        Yields:
            Chunks of LLM response
        """
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "provider": "Groq"
        }
