import boto3
import json
import logging

from botocore.exceptions import ClientError

## Instantiate Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

class FoundationModels:
    def __init__(self, bedrock_client) -> None:
        self.bedrock_client = bedrock_client

    def get_list(self):
        ## List all the foundation models deployed with Amazon Bedrock
        model_list = self.bedrock_client.list_foundation_models()
        ## Count Models
        counts = len(model_list["modelSummaries"])
        logger.info(f"Total Models- {counts}")

        ## Print Model Details
        logger.info(f"Models-\n {json.dumps(model_list, indent=2)}")
