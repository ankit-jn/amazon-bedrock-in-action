import json
import logging

## Instantiate Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

MODEL_ID_TITAN = "amazon.titan-embed-g1-text-02"
DEFAULT_PROMPT = "Why do we dream?"


class AmazonTitanEmbeddeing:
    """
    --> Amazon Titan Embedding models:

    1. amazon.titan-embed-g1-text-02:

    --> Request Structure: Json with following propertis

        {
            "body": string          ## Input data in the format specified in the content-type request header.
            "modelId": string       ## Identifier of the foundation model.
            "accept":  string       ## The desired MIME type of the inference body in the response.
            "contentType": string   ## The MIME type of the input data in the request
        }

        Request Body
        {
            "inputText": string
        }

    --> Response Structure: Json with following properties

        {
            "embedding": [float, float, ...],   ## An array that represents the vector of embeddings
            "inputTextTokenCount": int          ## The number of tokens in the input.
        }
    """

    def __init__(self, bedrock_client) -> None:
        self.bedrock_client = bedrock_client

    def prepare_input(self):
        self.model_id = (
            input(f"Please input modelId [{MODEL_ID_TITAN}]: ").strip()
            or MODEL_ID_TITAN
        )
        self.prompt = (
            input(f"Please input Question [{DEFAULT_PROMPT}]: ").strip()
            or DEFAULT_PROMPT
        )

    def process(self):
        """
        Generate a embeddings vector for a text input
        """
        ## Prepare the input for model invocation
        self.prepare_input()

        ## Prepare the input for FM invocation
        input = json.dumps(dict(inputText=self.prompt))

        ## Invoke the model
        output = self.bedrock_client.invoke_model(
            body=input,
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json",
        )

        ## Read the response
        response = json.loads(output["body"].read())

        embedding = response["embedding"]

        ## Print the embedding generated
        logger.info(embedding)
