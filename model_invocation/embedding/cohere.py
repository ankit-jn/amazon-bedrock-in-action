import json
import logging

## Instantiate Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

MODEL_ID_COHERE = "cohere.embed-english-v3"

INPUT_TYPE = "classification"
TRUNCATE_HANDLING = "NONE"


class CohereEmbeddeing:
    """
    --> Cohere Embedding models:

    1. cohere.embed-english-v3:

    --> Request Structure: Json with following propertis

        {
            "body": string          ## Input data in the format specified in the content-type request header.
            "modelId": string       ## Identifier of the foundation model.
            "accept":  string       ## The desired MIME type of the inference body in the response.
            "contentType": string   ## The MIME type of the input data in the request
        }

        Request Body
        {
            "texts":[string],                                                           ## Aray of text to embed
            "input_type": "search_document|search_query|classification|clustering",     ## Prepends special tokens to differentiate each type from one another.
            "truncate": "NONE|LEFT|RIGHT"                                               ## Specifies how the API handles inputs longer than the maximum token length
        }

    --> Response Structure: Json with following properties

        {
            "embeddings": [
                [ <array of 1024 floats> ]
            ],
            "id": string,
            "response_type" : "embeddings_floats,
            "texts": [string]
        }
    """

    def __init__(self, bedrock_client) -> None:
        self.bedrock_client = bedrock_client

    def prepare_input(self):
        self.model_id = (
            input("Please input modelId [cohere.embed-english-v3]: ").strip()
            or MODEL_ID_COHERE
        )
        self.input_type = (
            input("Please input input_type [classification]: ").strip() or INPUT_TYPE
        )
        self.truncate_handling = (
            input("Please input value for truncate [NONE]: ").strip()
            or TRUNCATE_HANDLING
        )

        self.prompt = (
            input("Please input Question [Why do we dream?]: ").strip()
            or "Why do we dream?"
        )

    def process(self):
        """
        Generate a embeddings vector for a text input
        """
        ## Prepare the input for model invocation
        self.prepare_input()

        ## Prepare the input for FM invocation
        input = json.dumps(
            dict(
                texts=[self.prompt],
                input_type=self.input_type,
                truncate=self.truncate_handling,
            )
        )

        ## Invoke the model
        output = self.bedrock_client.invoke_model(
            body=input,
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json",
        )

        ## Read the response
        response = json.loads(output["body"].read())

        ## Print the embedding generated
        logger.info(f"ID: {response.get('id')}")
        logger.info(f"Response type: {response.get('response_type')}")

        logger.info("Embeddings...")
        embedding = response["embeddings"]
        logger.info(f"Generated Embedding: {embedding}")

        logger.info("Texts...")
        texts = response["texts"]
        logger.info(f"Texts: {texts}")
