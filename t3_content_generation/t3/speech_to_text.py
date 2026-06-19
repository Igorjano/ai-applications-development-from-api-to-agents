import json

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST


# https://developers.openai.com/api/docs/guides/speech-to-text

class OpenAIClient:
    def __init__(self, api_key, endpoint):
        if not api_key:
            raise ValueError("API key not provided")
        self._api_key = api_key
        self._endpoint = endpoint

    def call(self, audio_file_path, **kwargs):
        headers = {'Authorization': 'Bearer ' + self._api_key}

        with open(audio_file_path, "rb") as audio_file:
            files = {"file": audio_file}

            response = requests.post(
                url=self._endpoint,
                headers=headers,
                files=files,
                data=kwargs
            )

        if response.status_code != 200:
            raise Exception(f"API request failed with error: {response.status_code} - {response.content}")

        print(json.dumps(response.json(), indent=2))


client = OpenAIClient(OPENAI_API_KEY, OPENAI_HOST + '/v1/audio/transcriptions')
client.call("audio_sample.mp3", model="whisper-1")
