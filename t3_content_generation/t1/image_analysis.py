import base64

from commons.constants import OPENAI_HOST
from t3_content_generation._openai_client import OpenAIClientT3


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main():
    endpoint = OPENAI_HOST + "/v1/chat/completions"
    client = OpenAIClientT3(endpoint=endpoint)

    image_name = "logo.png"
    base64_image = encode_image(image_name)

    img_urls = ["https://a-z-animals.com/media/2019/11/Elephant-male-1024x535.jpg", f"data:image/png;base64,{base64_image}"]

    img_content = [
        {"type": "image_url",
         "image_url":
             {"url": url}
         }
        for url in img_urls]

    client.call(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Generate poem based on these two images"},
                    *img_content
                ],
            }
        ],
        max_tokens=300,
    )


if __name__ == "__main__":
    main()
