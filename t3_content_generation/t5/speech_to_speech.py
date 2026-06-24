import base64
import json
from datetime import datetime

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST


# https://developers.openai.com/api/docs/guides/audio#add-audio-to-your-existing-application


def encode_audio(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


class OpenAIClient:
    def __init__(self, api_key, endpoint):
        if not api_key:
            raise ValueError("API key not provided")
        self._api_key = api_key
        self._endpoint = endpoint

    def call(self, **kwargs):
        headers = {
            'Authorization': 'Bearer ' + self._api_key,
            'Content-Type': 'application/json'
        }

        response = requests.post(
                url=self._endpoint,
                headers=headers,
                json=kwargs
            )

        if response.status_code != 200:
            raise Exception(f"API request failed with error: {response.status_code} - {response.content}")

        data = response.json()

        choices = data.get("choices", [])
        if choices:
            audio_data = choices[0].get("message", {}).get("audio", {}).get("data")

            if audio_data:
                audio_bytes = base64.b64decode(audio_data)
                output_file = "result_audio.mp3"

                with open(output_file, 'wb') as f:
                    f.write(audio_bytes)


client = OpenAIClient(OPENAI_API_KEY, OPENAI_HOST + '/v1/chat/completions')
client.call(
    model="gpt-audio",
    modalities=["text", "audio"],
    audio={"voice": "ballad", "format": "mp3"},
    messages=[
        {
          "role": "user",
          "content": [
            { "type": "input_audio", "input_audio": {"data": encode_audio("question.mp3"), "format": "mp3"}}
          ]
        }
      ]
)
