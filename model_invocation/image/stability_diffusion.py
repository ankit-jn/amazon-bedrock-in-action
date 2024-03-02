import json
import logging
import base64
from io import BytesIO
from PIL import Image

from utils.exception_handler import ImageException

## Instantiate Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Refer Latest documentation for Model Ids based on your use case
MODEL_ID_COMMAND = "stability.stable-diffusion-xl-v1"

# Inference Parameters Default Values
IMG_WIDTH = "1024"
IMG_HEIGHT = "1024"
CFG_SCALE = "7"
SEED = "0"
STEPS = "50"
STYLE_PRESET = "photographic"


class StabilityDiffusionImageGenerator:
    """
    --> Stability AI Diffusion Image models:

    1. SDXL 1.0 -> stability.stable-diffusion-xl-v1
    2. SDXL 0.8 -> stability.stable-diffusion-xl
    2. SDXL 0.8 -> stability.stable-diffusion-xl-v0


    --> Inference parameters: To control the Model Responses

        1. text_prompts:
        An array of text prompts to use for generation. Each element is a JSON object that contains a prompt
        and a weight for the prompt.

        2. width:
        Width of the image to generate, in pixels, in an increment divible by 64.
        The value must be one of 1024x1024, 1152x896, 1216x832, 1344x768, 1536x640, 640x1536, 768x1344, 832x1216, 896x1152.
        (default: 1024)
        
        3. height:
        Height of the image to generate, in pixels, in an increment divible by 64.
        The value must be one of 1024x1024, 1152x896, 1216x832, 1344x768, 1536x640, 640x1536, 768x1344, 832x1216, 896x1152.
        (default: 1024)

        4. cfg_scale: Determines how much the final image portrays the prompt. Use a lower number to increase
        randomness in the generation. (default to 7, range: 0-35)

        5. clip_guidance_preset:
        Enum: FAST_BLUE, FAST_GREEN, NONE, SIMPLE SLOW, SLOWER, SLOWEST.

        6. sampler:
        The sampler to use for the diffusion process. If this value is omitted, the model automatically selects
        an appropriate sampler for you.
        Enum: DDIM, DDPM, K_DPMPP_2M, K_DPMPP_2S_ANCESTRAL, K_DPM_2, K_DPM_2_ANCESTRAL, K_EULER, K_EULER_ANCESTRAL, K_HEUN K_LMS.

        7. samples:
        The number of image to generate. Currently Amazon Bedrock supports generating one image.

        8. seed:
        The seed determines the initial noise setting. Use the same seed and the same settings as a previous run to allow
        inference to create a similar image. If you don't set this value, or the value is 0, it is set as a random number.
        (default to 0, range: 0-4294967295)

        9. steps:
        Generation step determines how many times the image is sampled. More steps can result in a more accurate result.
        (default to 50, range: 0-150)

        10. style_preset:
        A style preset that guides the image model towards a particular style. This list of style presets is subject to change.
        Enum: 3d-model, analog-film, anime, cinematic, comic-book, digital-art, enhance, fantasy-art, isometric, line-art,
        low-poly, modeling-compound, neon-punk, origami, photographic, pixel-art, tile-texture.

    --> Request Structure: Json with following propertis

        {
            "body": string          ## Input data in the format specified in the content-type request header.
            "modelId": string       ## Identifier of the foundation model.
            "accept":  string       ## The desired MIME type of the inference body in the response.
            "contentType": string   ## The MIME type of the input data in the request
        }

        Request Body
        {
            "text_prompts": [
                {
                    "text": string,
                    "weight": float
                }
            ],
            "height": int,
            "width": int,
            "cfg_scale": float,
            "clip_guidance_preset": string,
            "sampler": string,
            "samples",
            "seed": int,
            "steps": int,
            "style_preset": string,
            "extras" :JSON object
        }



    --> Response Structure: Json with following properties

        {
            "result": string,
            "artifacts": [
                {
                    "seed": int,
                    "base64": string,
                    "finishReason": string
                }
            ]
        }

    """

    def __init__(self, bedrock_client) -> None:
        self.bedrock_client = bedrock_client

    def prepare_input(self):
        self.model_id = (
            input("Please input modelId [stability.stable-diffusion-xl-v1]: ").strip()
            or MODEL_ID_COMMAND
        )

        self.width = int(input("Please input width [1024]: ").strip() or IMG_WIDTH)
        self.height = int(input("Please input height [1024]: ").strip() or IMG_HEIGHT)

        self.cfg_scale = int(input("Please input cfg_scale [7]: ").strip() or CFG_SCALE)
        self.seed = int(input("Please input seed [0]: ").strip() or SEED)
        self.steps = int(input("Please input steps [50]: ").strip() or STEPS)
        self.style_preset = (
            input("Please input style_preset [photographic]: ").strip() or STYLE_PRESET
        )

        self.prompt = (
            input("Please input Question [A boy is playing with dog in the park.]: ").strip()
            or "A boy is playing with dog in the park."
        )

    def process(self):
        """
        Invoke Stability Diffusion Image Model
        """
        ## Collect user Inputs
        self.prepare_input()

        ### Prepare Input for the FM invocation
        input = json.dumps(
            dict(
                text_prompts=[dict(text=self.prompt)],
                width=self.width,
                height=self.height,
                cfg_scale=self.cfg_scale,
                seed=self.seed,
                steps=self.steps,
                style_preset=self.style_preset,
            )
        )

        ### Invoke Foundation Model
        output = self.bedrock_client.invoke_model(
            body=input,
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json",
        )

        ### Read Response
        response = json.loads(output["body"].read())

        for artifact in response["artifacts"]:
            finish_reason = artifact["finishReason"]
            if finish_reason == "ERROR" or finish_reason == "CONTENT_FILTERED":
                raise ImageException(f"Error in Image Generation: {finish_reason}")

            base64_img = artifact["base64"]
            base64_bytes = base64_img.encode("ascii")
            img_bytes = base64.b64decode(base64_bytes)

            image = Image.open(BytesIO(img_bytes))
            image.save("generated_image.png")
