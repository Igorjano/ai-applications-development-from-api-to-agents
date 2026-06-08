import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class CustomAnthropicAIClient(AIClient):
    """
    Custom HTTP client for Anthropic's Claude API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with Claude's API directly
    and handle its Server-Sent Events (SSE) streaming format.
    """

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no content blocks.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            Requires 'x-api-key' header and 'anthropic-version' header.
            Claude's API returns content as an array of content blocks.
            The response is printed to stdout before being returned.
        """
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

        messages_input = [msg.to_dict() for msg in messages]

        data = {
            "model": self._model_name,
            "messages": messages_input,
            "max_tokens": 1024,
            "system": self._system_prompt
        }

        response = requests.post(self._endpoint, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"API request failed with error: {response.status_code} - {response.text}")

        data = response.json()
        content = data.get("content", [])
        if not content:
            raise ValueError("API response contains no content")

        text = "".join(b.get("text", "") for b in content)
        print(f"✨: {text}")

        return Message(role=Role.ASSISTANT, content=text)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed using Anthropic's SSE format, with text deltas
        printed immediately as they arrive.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all deltas are received.

        Note:
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
            Listens for 'content_block_delta' events with 'text_delta' type.
            Stops processing when 'message_stop' event is received.
            Each delta is printed to stdout as it arrives.
        """
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

        messages_input = [msg.to_dict() for msg in messages]

        data = {
            "model": self._model_name,
            "messages": messages_input,
            "max_tokens": 1024,
            "system": self._system_prompt,
            "stream": True
        }

        chunks_list = []
        async with aiohttp.ClientSession() as session:
            async with session.post(self._endpoint, headers=headers, json=data) as response:
                if response.status != 200:
                    raise Exception(f"API request failed with error: {response.status} - {response.text}")

                async for line in response.content:
                    line_str = line.decode("utf-8").strip()

                    if line_str.startswith("data: "):
                        data = line_str[len("data: "):]
                        parse_data = json.loads(data)

                        if parse_data.get("type", "") == "content_block_delta":
                            text_chunk = parse_data.get("delta", {}).get("text", "")
                            if text_chunk:
                                print(text_chunk, end="")
                                chunks_list.append(text_chunk)

                print()

        return Message(role=Role.ASSISTANT, content="".join(chunks_list))
