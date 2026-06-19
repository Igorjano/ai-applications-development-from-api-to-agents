import base64
from datetime import datetime

from commons.constants import OPENAI_HOST
from t3_content_generation._openai_client import OpenAIClientT3


# https://developers.openai.com/api/reference/resources/images/methods/generate
# ---
# Request:
# curl -X POST "https://api.openai.com/v1/images/generations" \
#     -H "Authorization: Bearer $OPENAI_API_KEY" \
#     -H "Content-type: application/json" \
#     -d '{
#         "model": "gpt-image-2",
#         "prompt": "smiling catdog."
#     }'
# Response:
# {
#   "created": 1699900000,
#   "data": [
#     {
#       "b64_json": Qt0n6ArYAEABGOhEoYgVAJFdt8jM79uW2DO...,
#     }
#   ]
# }

def main():
    endpoint = OPENAI_HOST + '/v1/images/generations'
    client = OpenAIClientT3(endpoint=endpoint)
    output_image_path = 'smiling_catdog.png'

    response = client.call(
        model='gpt-image-2',
        prompt='Smiling catdog',
    )

    image_base64 = response.get('data', [])[0].get('b64_json', '')
    if image_base64:
        image_bytes = base64.b64decode(image_base64)

        with open(output_image_path, 'wb') as image_file:
            image_file.write(image_bytes)


if __name__=='__main__':
    main()
