import base64
from datetime import datetime

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST


# https://developers.openai.com/api/reference/resources/images/methods/edit
# ---
# Request (multipart/form-data, NOT json):
# curl -X POST "https://api.openai.com/v1/images/edits" \
#     -H "Authorization: Bearer $OPENAI_API_KEY" \
#     -F "model=gpt-image-1" \
#     -F "image=@logo.png" \
#     -F "prompt=Add magical sparkles and glowing aura around the logo"
# Response:
# {
#   "created": 1699900000,
#   "data": [
#     {
#       "b64_json": "Qt0n6ArYAEABGOhEoYgVAJFdt8jM79uW2DO..."
#     }
#   ]
# }

def main():
    image_path = 'logo.png'
    output_image_path = 'magic_logo.png'

    with open(image_path, "rb") as image_file:
        url = OPENAI_HOST + '/v1/images/edits'

        headers = {
            'Authorization': 'Bearer ' + OPENAI_API_KEY,
        }

        data = {
            'model': 'gpt-image-2',
            'prompt': 'Add some "magic" to the logo — bright sparkles and a little glow with shooting stars',
        }

        files = {'image': (image_path, image_file, 'image/png')}

        response = requests.post(url, headers=headers, data=data, files=files)

    if response.status_code != 200:
        raise Exception(f"API request failed with error: {response.status_code} - {response.content()}")

    image_base64 = response.json().get('data', [])[0].get('b64_json', '')
    if image_base64:
        image_bytes = base64.b64decode(image_base64)

        with open(output_image_path, 'wb') as image_file:
            image_file.write(image_bytes)


if __name__ == '__main__':
    main()
