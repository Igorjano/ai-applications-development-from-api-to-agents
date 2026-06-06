import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.openai.base import BaseOpenAIClient


class CustomOpenAIResponsesClient(BaseOpenAIClient):
    """
    Custom HTTP client for OpenAI Responses API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with the Responses API directly
    and handle its unique event-based streaming format.
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
            ValueError: If the API response contains no output text.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            Uses the Responses API format with 'instructions' and 'input' parameters.
            The response is printed to stdout before being returned.
        """
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }

        input_messages = [msg.to_dict() for msg in messages]

        data = {
            "model": self._model_name,
            "instructions": self._system_prompt,
            "input": input_messages,
            "max_output_tokens": 1024
        }

        response = requests.post(self._endpoint, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"API request failed with error: {response.status_code} - {response.text}")

        data = response.json()
        output = data.get("output", [])
        if not output:
            raise ValueError("API response contains no output")

        text = output[0].get("content", [])[0].get("text", "")
        print(f"✨: {text}")

        return Message(role=Role.ASSISTANT, content=text)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with event-based streaming.

        The Responses API uses a different SSE format than Chat Completions,
        with explicit event types and data fields.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The complete AI response message after all deltas are received.

        Note:
            Uses event-based Server-Sent Events (SSE) format.
            Listens for 'response.output_text.delta' events to build the response.
            Each line with "event: " specifies the event type, followed by "data: " with the payload.
        """
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }

        input_messages = [msg.to_dict() for msg in messages]

        data = {
            "model": self._model_name,
            "instructions": self._system_prompt,
            "input": input_messages,
            "max_output_tokens": 1024,
            "stream": True
        }

        chunks_list = []
        print("✨: ", end="")

        async with aiohttp.ClientSession() as session:
            async with session.post(self._endpoint, headers=headers, json=data) as response:
                if response.status != 200:
                    raise Exception(f"API request failed with error: {response.status} - {await response.text()}")

                async for line in response.content:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data"):
                        data = line_str[len("data: "):]
                        parsed_data = json.loads(data)

                        if parsed_data.get("type", "") == "response.output_text.delta":
                            text_chunk = parsed_data.get("delta", "")
                            if text_chunk:
                                print(text_chunk, end="")
                                chunks_list.append(text_chunk)
            print()

        return Message(role=Role.ASSISTANT, content="".join(chunks_list))
