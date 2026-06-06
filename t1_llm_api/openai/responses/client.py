from openai import OpenAI, AsyncOpenAI

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.openai.base import BaseOpenAIClient


class OpenAIResponsesClient(BaseOpenAIClient):
    """
    Client for OpenAI Responses API using the official SDK.

    This implementation uses the official OpenAI Python library to interact
    with the Responses API, which uses 'instructions' instead of system messages
    and 'input' instead of messages.

    Attributes:
        _client (OpenAI): Synchronous OpenAI client instance.
        _async_client (AsyncOpenAI): Asynchronous OpenAI client instance.
        Inherits all other attributes from BaseOpenAIClient.
    """

    def __init__(self, endpoint: str, model_name: str, system_prompt: str, api_key: str):
        """
        Initialize the OpenAI Responses client with SDK.

        Args:
            endpoint (str): The OpenAI API endpoint (for compatibility, not used by SDK).
            model_name (str): The OpenAI model to use (e.g., 'gpt-5').
            system_prompt (str): The instruction to guide the model's behavior.
            api_key (str): The OpenAI API key for authentication.
        """
        super().__init__(endpoint, model_name, system_prompt, api_key)
        self._model_name = model_name
        self._system_prompt = system_prompt
        self._client = OpenAI(api_key=api_key)
        self._async_client = AsyncOpenAI(api_key=api_key)

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response from OpenAI's Responses API.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The AI's response message.

        Note:
            Uses the Responses API format with 'instructions' and 'input' parameters.
            The response is printed to stdout before being returned.
        """
        input_messages = [msg.to_dict() for msg in messages]

        response = self._client.responses.create(
            model=self._model_name,
            instructions=self._system_prompt,
            input=input_messages,
            max_output_tokens=1024
        )

        text = response.output_text
        print(f"✨: {text}")

        return Message(role=Role.ASSISTANT, content=text)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response from OpenAI's Responses API.

        The response is streamed using event-based streaming, with each delta
        printed immediately as it arrives.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The complete AI response message after all deltas are received.

        Note:
            Uses the Responses API streaming format with event types.
            Listens for 'response.output_text.delta' events to build the response.
        """
        input_messages = [msg.to_dict() for msg in messages]
        chunks_list = []
        print("✨: ", end="")

        async with self._async_client.responses.stream(
            model=self._model_name,
            instructions=self._system_prompt,
            input=input_messages,
            max_output_tokens=1024
        ) as stream:
            async for event in stream:
                if event.type == "response.output_text.delta":
                    print(event.delta, end="")
                    chunks_list.append(event.delta)

            print()

        return Message(role=Role.ASSISTANT, content="\n".join(chunks_list))
