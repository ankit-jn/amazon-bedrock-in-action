import json
import logging

## Instantiate Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

MODEL_ID_TITAN = "amazon.titan-embed-g1-text-02"


class AmazonTitanEmbeddeing:

    def __init__(self, bedrock_client) -> None:
        self.bedrock_client = bedrock_client

    def prepare_input(self):
        self.model_id = (
            input(
                "Please input modelId [amazon.amazon.titan-embed-g1-text-02]: "
            ).strip()
            or MODEL_ID_TITAN
        )
        self.prompt = (
            input("Please input Question [Why do we dream?]: ").strip()
            or "Why do we dream?"
        )

    def process(self):
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
