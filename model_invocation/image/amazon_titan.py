import json
import logging
import base64
from io import BytesIO
from PIL import Image

from utils.exception_handler import BedrockException

## Instantiate Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Refer Latest documentation for Model Ids based on your use case
MODEL_ID_TITAN = "amazon.titan-image-generator-v1"

# Inference Parameters Default Values
IMAGE_COUNTS = "3"
QUALITY = "standard"
CFG_SCALE = "7"
SEED = "0"
IMG_WIDTH = "1024"
IMG_HEIGHT = "1024"
NEGATIVE_TEXT = ""


class AmazonTitanImageGenerator:
    """
    --> Amazon Titan Image models:

    1. amazon.titan-image-generator-v1


    --> Inference parameters: To control the Model Responses

        1. text:
        A text prompt to use for image generation.

        2. negativeText
        A text prompt to define what not to include in the image.

        3. numberOfImages:
        Number of images to be generated (range: 1-5, default 3)

        4. quality:
        Quality of generated images. Values: standard,premium (Default: standard)

        5. width:
        Width of the image to generate, in pixels, in an increment divible by 64.
        The value must be one of 1024x1024, 1152x896, 1216x832, 1344x768, 1536x640, 640x1536, 768x1344, 832x1216, 896x1152.
        (default: 1024)

        6. height:
        Height of the image to generate, in pixels, in an increment divible by 64.
        The value must be one of 1024x1024, 1152x896, 1216x832, 1344x768, 1536x640, 640x1536, 768x1344, 832x1216, 896x1152.
        (default: 1024)

        7. cfg_scale:
        Scale for classifier-free guidance. It determines how much the final image portrays the prompt. Use a lower number to
        increase randomness in the generation. (default to 8, range: 1.1-10)

        8. seed:
        The seed to use for reproducibility. It determines the initial noise setting. Use the same seed and the same settings
        as a previous run to allow inference to create a similar image. If you don't set this value, or the value is 0,
        it is set as a random number. (default to 0, range: 0-214783647)



    --> Request Structure: Json with following propertis

        {
            "body": string          ## Input data in the format specified in the content-type request header.
            "modelId": string       ## Identifier of the foundation model.
            "accept":  string       ## The desired MIME type of the inference body in the response.
            "contentType": string   ## The MIME type of the input data in the request
        }

        Request Body
        {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": "string",
                "negativeText": "string"
            },
            "imageGenerationConfig": {
                "numberOfImages": int,
                "height": int,
                "width": int,
                "cfgScale": float,
                "seed": int
            }
        }



    --> Response Structure: Json with following properties

        {
            "images": [
                "base64-encoded string",
                ...
            ],
            "error": "string"
            }

    """

    def __init__(self, bedrock_client) -> None:
        self.bedrock_client = bedrock_client

    def prepare_input(self):
        self.model_id = (
            input("Please input modelId [amazon.titan-image-generator-v1]: ").strip()
            or MODEL_ID_TITAN
        )

        self.img_counts = (
            int(input("Please input numberOfImages [3]: ").strip() or IMAGE_COUNTS)
        )
        self.quality = input("Please input quality [standard]: ").strip() or QUALITY
        self.width = int(input("Please input width [1024]: ").strip() or IMG_WIDTH)
        self.height = int(input("Please input height [1024]: ").strip() or IMG_HEIGHT)

        self.cfg_scale = float(
            input("Please input cfgScale [8.0]: ").strip() or CFG_SCALE
        )
        self.seed = int(input("Please input seed [0]: ").strip() or SEED)

        self.prompt = (
            input(
                "Please input text [A boy is playing with dog in the park.]: "
            ).strip()
            or "A boy is playing with dog in the park."
        )
        self.negative_text = (
            input("Please input negativeText [None]: ").strip() or NEGATIVE_TEXT
        )

    def process(self):
        """
        Invoke Amazon Titan Image Model
        """
        ## Collect user Inputs
        self.prepare_input()

        ### Prepare Input for the FM invocation
        if self.negative_text == "":
            textToImageParams = dict(text=self.prompt)
        else:
            textToImageParams = dict(text=self.prompt, negativeText=self.negative_text)

        input = json.dumps(
            dict(
                taskType="TEXT_IMAGE",
                textToImageParams=textToImageParams,
                imageGenerationConfig=dict(
                    numberOfImages=self.img_counts,
                    quality=self.quality,
                    width=self.width,
                    height=self.height,
                    cfgScale=self.cfg_scale,
                    seed=self.seed,
                ),
            )
        )
        print(input)

        ### Invoke Foundation Model
        output = self.bedrock_client.invoke_model(
            body=input,
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json",
        )

        ### Read Response
        response = json.loads(output["body"].read())

        error = response.get("error")
        if error is not None:
            raise BedrockException(f"Image Generation Error: {error}")

        images = [
            Image.open(BytesIO(base64.b64decode(base64_image)))
            for base64_image in response.get("images")
        ]
        num_image = 1
        for image in images:
            image.save(f"generated_image-{num_image}.png")
            num_image = num_image + 1
