import json
import logging
from utils.exception_handler import BedrockException

## Instantiate Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

# Refer Latest documentation for Model Ids based on your use case
MODEL_ID_TITAN = "amazon.titan-text-express-v1"

# Inference Parameters Default Values
TEMPERATURE = "0.9"
TOP_P = "1.0"
MAX_TOKEN_COUNT = "512"
STOP_SEQUENCES = []

class AmazonTextModel:
    """
    --> Amazon text models:

    1. titan-text-lite-v1:
    Amazon Titan Text Lite is a cost efective and light weight efficient model ideal for fine-tuning for English-language tasks, including 
    like summarization and copywriting, where customers want a smaller, more cost-effective model

    2. titan-text-express-v1: 
    Amazon Titan Text Express has a context length of up to 8,000 tokens, making it well-suited for a wide range of advanced, 
    general language tasks such as open-ended text generation and conversational chat, code generation, table creation etc.


    --> Inference parameters: To control the Model Responses

    A. Randomness and Diversity ###

        1. temperature:
        Modulates the probability density function for the next tokens, implementing the temperature sampling technique.
        This parameter can be used to deepen or flatten the density function curve.
        A lower value results in a steeper curve and more deterministic responses,
        whereas a higher value results in a flatter curve and more random (creative and different) responses for the same prompt.
        (float, defaults to 0, max value is 1.5)

        2. topP:
        Top P controls token choices, based on the probability of the potential choices. If you set Top P below 1.0,
        the model considers only the most probable options and ignores less probable options.
        The result is more stable and repetitive completions. (defaults to 0.9, max value is 1.0)

    B. Length

        1. maxTokenCount:
        Configures the max number of tokens to use in the generated response. (int, defaults to 512)

        2. stopSequences:
        Used to make the model stop at a desired point, such as the end of a sentence or a list.
        The returned response will not contain the stop sequence. (defaults to blank)

    --> Request Structure: Json with following propertis

        body: Input data in the format specified in the content-type request header.
        modelId: Identifier of the foundation model.
        accept:  The desired MIME type of the inference body in the response.
        contentType: The MIME type of the input data in the request

    --> Response Structure: Json with following properties

        body: Inference response from the model in the format specified in the content-type header field.
        contentType: The MIME type of the inference result.

    """
    def __init__(self, bedrock_client) -> None:
        self.bedrock_client = bedrock_client

    def prepare_input(self):
        self.model_id = input('Please input modelId [amazon.titan-text-express-v1]: ').strip() or MODEL_ID_TITAN

        self.temperature = float(input('Please input temperature [0.9]: ').strip() or TEMPERATURE)
        self.top_p = float(input('Please input topP [1.0]: ').strip() or TOP_P)
        self.max_token_input = int(input('Please input maxTokenCount [512]: ').strip() or MAX_TOKEN_COUNT)
        self.stop_sequences = (input('Please comma seperated input stopSequences [None]: ').strip() or STOP_SEQUENCES)
        if type(self.stop_sequences) == str:
            self.stop_sequences = self.stop_sequences.split(",")
        
        self.prompt = input('Please input Question [Why do we dream?]: ').strip() or "Why do we dream?"

    def invoke_titan_model(self):
        """
        Invoke Amazon Titan Text Model
        """
        ## Collect user Inputs
        self.prepare_input()

        ### Prepare Input for the FM invocation
        input = json.dumps(
            dict(
                inputText=self.prompt,
                textGenerationConfig=dict(
                    maxTokenCount=self.max_token_input,
                    stopSequences=self.stop_sequences,
                    temperature=self.temperature,
                    topP=self.top_p,
                ),
            )
        )

        ### Invoke Foundation Model
        output = self.bedrock_client.invoke_model(
            body=input,
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json",
        )

        ### Process Response
        response = json.loads(output["body"].read())

        error = response.get("error")
        if error is not None:
            raise BedrockException(f"Text Generation Error: {error}")
        
        logger.info(f"Input text Token Count: {response["inputTextTokenCount"]}")
        for result in response["results"]:
            logger.info(f"Token Count: {result["tokenCount"]}")
            logger.info(f"Output text: {result["outputText"]}")
            logger.info(f"Completion Reason: {result["completionReason"]}")
