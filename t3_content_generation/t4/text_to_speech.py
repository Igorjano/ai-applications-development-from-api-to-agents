import json
from datetime import datetime

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST


class Voice:
    alloy: str = 'alloy'
    ash: str = 'ash'
    ballad: str = 'ballad'
    coral: str = 'coral'
    echo: str = 'echo'
    fable: str = 'fable'
    nova: str = 'nova'
    onyx: str = 'onyx'
    sage: str = 'sage'
    shimmer: str = 'shimmer'


# https://developers.openai.com/api/docs/guides/text-to-speech
# Request:
# curl https://api.openai.com/v1/audio/speech \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "gpt-4o-mini-tts",
#     "input": "Why can't we say that black is white?",
#     "voice": "coral",
#     "instructions": "Speak in a cheerful and positive tone."
#   }' \
# Response:
#   bytes with audio


output_file_path = 'result_audio.mp3'


class OpenAIClient:
    def __init__(self, api_key, endpoint):
        if not api_key:
            raise ValueError('API key not provided')
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

        with open(output_file_path, 'wb') as mp3_file:
            mp3_file.write(response.content)


client = OpenAIClient(OPENAI_API_KEY, OPENAI_HOST + '/v1/audio/speech')
client.call(
    model="gpt-4o-mini-tts",
    input="Why can't we say that black is white?",
    voice=Voice.ash,
    instructions="Speak in a cheerful and positive tone."
)
