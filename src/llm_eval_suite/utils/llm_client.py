import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from .logging_utils import get_logger

logger = get_logger(__name__)

load_dotenv()


class LLMClient:
    """Thin wrapper around OpenAI / Anthropic clients.

    This class focuses on:
    - Choosing the correct provider.
    - Building messages from prompt templates.
    - Exposing a simple .generate() method for the eval harness.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        temperature: float = 0.0,
    ) -> None:
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature

        if self.provider == "openai":
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set in environment.")
            self.client = OpenAI(api_key=api_key)

        elif self.provider == "anthropic":
            import anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set in environment.")
            self.client = anthropic.Anthropic(api_key=api_key)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a completion given system + user prompts.

        Returns the raw text content from the model.
        """
        logger.info("Calling LLM provider=%s model=%s", self.provider, self.model_name)

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            # Simple Anthropic example
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            # anthropic returns a list of content blocks
            parts = []
            for block in response.content:
                if block.type == "text":
                    parts.append(block.text)
            return "".join(parts)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
