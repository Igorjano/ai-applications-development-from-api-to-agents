import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class CustomGeminiAIClient(AIClient):
    """
    Custom HTTP client for Google Gemini API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with Gemini's API directly
    and handle its Server-Sent Events (SSE) streaming format.
    """

    def _to_gemini_contents(self, messages: list[Message]) -> list[dict]:
        """
        Convert Message objects to Gemini Content format.

        Gemini uses a different role naming convention where AI messages use
        the role "model" instead of "assistant".

        Args:
            messages (list[Message]): The conversation messages to convert.

        Returns:
            list[dict]: Messages in Gemini's Content format.
        """
        contents = []
        for msg in messages:
            role = msg.role
            contents.append(
                {
                    "role": role,
                    "parts": [
                        {
                            "text": msg.content
                        }
                    ]
                }
            )

        return contents

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no candidates.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            The URL is constructed by appending ':generateContent' to the model endpoint.
            Uses 'x-goog-api-key' header for authentication.
            Response candidates contain content parts that are concatenated.
        """
        headers = {
            'x-goog-api-key': self._api_key,
            'Content-Type': 'application/json'
        }

        data = {
            "system_instruction": {
                "parts": [
                    {"text": self._system_prompt}
                ]
            },
            "contents": self._to_gemini_contents(messages),
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", 1024)
            }
        }

        url = f"{self._endpoint}/{self._model_name}:generateContent"

        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"API request failed with error: {response.status_code} - {response.text}")

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("API response contains no candidates")

        parts = data["candidates"][0].get("content", {}).get("parts", {})
        text = "".join(part.get("text", "") for part in parts)
        print(text)

        return Message(role=Role.ASSISTANT, content=text)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed using Gemini's SSE format, with text chunks
        printed immediately as they arrive.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            The URL is constructed with ':streamGenerateContent?alt=sse' endpoint.
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
            Each SSE chunk contains candidates with content parts.
            Each text chunk is printed to stdout as it arrives.
        """
        headers = {
            'x-goog-api-key': self._api_key,
            'Content-Type': 'application/json'
        }

        data = {
            "system_instruction": {
                "parts": [
                    {"text": self._system_prompt}
                ]
            },
            "contents": self._to_gemini_contents(messages),
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", 1024)
            }
        }

        url = f"{self._endpoint}/{self._model_name}:streamGenerateContent?alt=sse"

        chunks_list = []

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status != 200:
                    raise Exception(f"API request failed with error: {response.status} - {await response.text()}")

                async for line in response.content:
                    line_str = line.decode("utf-8").strip()

                    if line_str.startswith("data: "):
                        data = line_str[len("data: "):]
                        parse_data = json.loads(data)
                        candidates = parse_data.get("candidates", [])

                        if not candidates:
                            raise ValueError("API response contains no candidates")

                        parts = candidates[0].get("content", {}).get("parts", {})
                        for part in parts:
                            text_chunk = part.get("text", "")
                            if text_chunk:
                                print(text_chunk, end="")
                                chunks_list.append(text_chunk)
                print()

            return Message(role=Role.ASSISTANT, content=''.join(chunks_list))
