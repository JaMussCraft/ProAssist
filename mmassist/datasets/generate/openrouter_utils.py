import os
import requests
import json
from typing import List, Tuple, Dict, Any
import time
from dataclasses import dataclass


@dataclass
class OpenRouterConfig:
    api_key: str
    model: str = "google/gemini-2.5-flash"
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: int = 4096
    temperature: float = 0.5
    top_p: float = 0.95


class OpenRouterGenerator:
    def __init__(self, config: OpenRouterConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/JaMussCraft/ProAssist",
            "X-Title": "ProAssist Dialog Generation"
        }
    
    @classmethod
    def build(cls, model_id: str = "anthropic/claude-3.5-sonnet", api_key: str = None):
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key is None:
                raise ValueError("OPENROUTER_API_KEY environment variable must be set")
        
        config = OpenRouterConfig(
            api_key=api_key,
            model=model_id,
        )
        return cls(config)
    
    def _make_request(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Make a single request to OpenRouter API"""
        data = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }
        
        try:
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=self.headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            raise
        except KeyError as e:
            print(f"Unexpected response format: {e}")
            print(f"Response: {response.text}")
            raise
    
    def generate(self, inputs: List[Tuple[str, str]], **kwargs) -> List[str]:
        """
        Generate responses for a single conversation.
        
        Args:
            inputs: List of (role, content) tuples for the conversation
            **kwargs: Additional parameters for the API call
            
        Returns:
            List with a single generated response
        """
        messages = [{"role": role, "content": content} for role, content in inputs]
        
        # Add retry logic for rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._make_request(messages, **kwargs)
                return [response]
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    print(f"Rate limited, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise
        
        raise Exception(f"Failed after {max_retries} retries")
    
    def batch_generate(self, inputs: List[List[Tuple[str, str]]], **kwargs) -> List[List[str]]:
        """
        Generate responses for multiple conversations.
        
        Args:
            inputs: List of conversations, each conversation is a list of (role, content) tuples
            **kwargs: Additional parameters for the API call
            
        Returns:
            List of lists, each containing a single generated response
        """
        results = []
        
        for i, conversation in enumerate(inputs):
            print(f"Processing conversation {i+1}/{len(inputs)}")
            
            # Add delay between requests to avoid rate limiting
            if i > 0:
                time.sleep(1)
            
            result = self.generate(conversation, **kwargs)
            results.append(result)
        
        return results


# For backward compatibility with existing code structure
class LLMGenerator:
    def __init__(self, openrouter_generator: OpenRouterGenerator):
        self.openrouter_generator = openrouter_generator
        # Default sampling args compatible with the existing interface
        self.default_sampling_args = dict(
            n=1, temperature=0.5, top_p=0.95, max_tokens=4096
        )
    
    @classmethod
    def build(cls, model_id: str, number_gpus: int = None, local_rank: int = None):
        """Build LLMGenerator with OpenRouter backend"""
        # Ignore GPU-related parameters since we're using API
        openrouter_gen = OpenRouterGenerator.build(model_id=model_id)
        return cls(openrouter_gen)
    
    def generate(self, inputs: List[Tuple[str, str]], **kwargs) -> List[str]:
        """Generate responses maintaining compatibility with existing interface"""
        # Filter out vLLM-specific parameters
        filtered_kwargs = {}
        if "temperature" in kwargs:
            filtered_kwargs["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            filtered_kwargs["top_p"] = kwargs["top_p"]
        if "max_tokens" in kwargs:
            filtered_kwargs["max_tokens"] = kwargs["max_tokens"]
        
        return self.openrouter_generator.generate(inputs, **filtered_kwargs)
    
    def batch_generate(self, inputs: List[List[Tuple[str, str]]], **kwargs) -> List[List[str]]:
        """Batch generate responses maintaining compatibility with existing interface"""
        # Filter out vLLM-specific parameters
        filtered_kwargs = {}
        if "temperature" in kwargs:
            filtered_kwargs["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            filtered_kwargs["top_p"] = kwargs["top_p"]
        if "max_tokens" in kwargs:
            filtered_kwargs["max_tokens"] = kwargs["max_tokens"]
        
        return self.openrouter_generator.batch_generate(inputs, **filtered_kwargs)