import json
import logging
from utils.exception_handler import BedrockException

## Instantiate Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

# Refer Latest documentation for Model Ids based on your use case
MODEL_ID_LLAMA = "meta.llama2-13b-chat-v1"

# Inference Parameters Default Values
TEMPERATURE = "0.5"
TOP_P = "0.9"
MAX_GEN_LEN = "512"

class MetaTextModel:
    """
    --> Meta text models:
    Llama 2 is a high-performance, auto-regressive language model designed for developers. 
    It uses an optimized transformer architecture and pretrained models are trained on 2 trillion tokens with a 4k context length.
    It is intended for commercial and research use in English. Fine-tuned chat models are intended for chat based applications.

    1. llama2-70b-chat-v1: 70 billion parameter base model
    2. llama2-13b-chat-v1: 13 billion parameter base model.


    --> Inference parameters: To control the Model Responses

    A. Randomness and Diversity ###

        1. temperature:
        Modulates the probability density function for the next tokens, implementing the temperature sampling technique.
        This parameter can be used to deepen or flatten the density function curve.
        A lower value results in a steeper curve and more deterministic responses,
        whereas a higher value results in a flatter curve and more random (creative and different) responses for the same prompt.
        (float, defaults to 0.5, max value is 1.0)

        2. topP:
        Top P controls token choices, based on the probability of the potential choices. If you set Top P below 1.0,
        the model considers only the most probable options and ignores less probable options.
        The result is more stable and repetitive completions. (defaults to 0.9, max value is 1.0)

    B. Length

        1. max_gen_len:
        Configures the max number of tokens to use in the generated response. (int, defaults to 512, range: 1-2048)

    --> Request Structure: Json with following propertis

        {
            "body": string          ## Input data in the format specified in the content-type request header.
            "modelId": string       ## Identifier of the foundation model.
            "accept":  string       ## The desired MIME type of the inference body in the response.
            "contentType": string   ## The MIME type of the input data in the request
        }

        Request Body
        {
            "prompt": string,
            "temperature": float,
            "top_p": float,
            "max_gen_len": int
        }

    --> Response Structure: Json with following properties

        ## Plain Response
        {
            "generation": "\n\n<response>",
            "prompt_token_count": int,
            "generation_token_count": int,
            "stop_reason" : string
        }

    """
    def __init__(self, bedrock_client) -> None:
        self.bedrock_client = bedrock_client

    def prepare_input(self):
        self.model_id = input('Please input modelId [meta.llama2-13b-chat-v1]: ').strip() or MODEL_ID_LLAMA

        self.temperature = float(input('Please input temperature [0.5]: ').strip() or TEMPERATURE)
        self.top_p = float(input('Please input top_p [0.9]: ').strip() or TOP_P)
        self.max_gen_len = int(input('Please input max_gen_len [512]: ').strip() or MAX_GEN_LEN)
        
        self.prompt = input('Please input Question [Why do we dream?]: ').strip() or "Why do we dream?"

    def process(self, streaming = False):

        ## Prepare Input for the FM invocation
        self.prepare_input()

        ## Prepare Input for the FM invocation
        input  = json.dumps(dict(
            prompt=self.prompt,
            temperature = self.temperature,
            top_p = self.top_p,
            max_gen_len = self.max_gen_len,
        ))

        ## Invoke Foundation Model
        if not streaming:
            output = self.bedrock_client.invoke_model(
                body = input,
                modelId = self.model_id,
                accept = "application/json",
                contentType = "application/json"
            )

            ## Read Response
            response = json.loads(output.get("body").read())
            logger.info(f"Generation: {response.get("generation")}")
        
        else:
            output = self.bedrock_client.invoke_model_with_response_stream(
                body = input,
                modelId = self.model_id,
                accept = "application/json",
                contentType = "application/json"
            )

            ## Read Response
            response_stream = output.get("body")

            for event in response_stream:
                chunk = event.get("chunk")
                data = json.loads(chunk.get("bytes").decode())
                text = data.get("generation")
                if '\n' == text:
                    print('')
                    continue
                print(text, end='')
