import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.openai.base import BaseOpenAIClient


class CustomOpenAIClient(BaseOpenAIClient):
    """
    Custom HTTP client for OpenAI Chat Completions API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, providing more control over the HTTP layer and demonstrating
    how to interact with the API directly.
    """

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no choices.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            The system prompt is automatically prepended to the messages.
            The response is printed to stdout before being returned.
        """
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }

        messages = [
            {"role": "developer", "content": self._system_prompt},
            *[msg.to_dict() for msg in messages]
        ]

        data = {
            "model": self._model_name,
            "messages": messages,
            "max_completion_tokens": 1024
        }

        response = requests.post(self._endpoint, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"API request failed with error: {response.status_code} - {response.text}")

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("API response contains no choices")

        content = choices[0].get("message", {}).get("content", "")
        print(content)

        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed token-by-token using OpenAI's SSE format,
        with each chunk printed immediately as it arrives.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            The system prompt is automatically prepended to the messages.
            Each token is printed to stdout as it arrives.
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
        """
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }

        messages = [
            {"role": "developer", "content": self._system_prompt},
            *[msg.to_dict() for msg in messages]
        ]

        data = {
            "model": self._model_name,
            "messages": messages,
            "max_completion_tokens": 1024,
            "stream": True
        }

        chunks_list = []
        async with aiohttp.ClientSession() as session:
            async with session.post(self._endpoint, headers=headers, json=data) as response:
                if response.status != 200:
                    raise Exception(f"API request failed with error: {response.status} - {await response.text()}")

                async for line in response.content:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        data = line_str[len("data: "):]

                        if data != "[DONE]":
                            parse_data = json.loads(data)
                            choices = parse_data.get("choices", [])

                            if not choices:
                                raise ValueError("API response contains no choices")

                            content_chunk = choices[0].get("delta", {}).get("content", "")
                            if content_chunk:
                                print(content_chunk, end="")
                                chunks_list.append(content_chunk)
                print()

        return Message(role=Role.ASSISTANT, content=''.join(chunks_list))
