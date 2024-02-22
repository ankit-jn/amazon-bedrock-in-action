import boto3
from botocore.exceptions import ClientError
import logging

from model_invocation.amazon_text_model import AmazonTextModel
from list_models import FoundationModels

from utils.exception_handler import BedrockException

## Instantiate Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

## Creating session with AWS profile
session = boto3.Session(profile_name="bedrock-profile")

# bedrock – Contains runtime plane APIs for making inference requests for models hosted in Amazon Bedrock
control_client = session.client("bedrock")

# bedrock-runtime – Contains runtime plane APIs for making inference requests for models hosted in Amazon Bedrock
runtime_client = session.client("bedrock-runtime")

def choice_option():
    print("------------ Gen AI Hands-on ------------")
    print("1. List all the models")
    print("2. Test Amazon Titan Text Model")
    print("3. Test Anthropic Claude Text Model")
    print("4. Exit")
    valid = False
    while not valid:
        choice = input('Please select option: ').strip()
        if choice.isnumeric():
            valid = True
            choice = int(choice)
        else:
            print("Looks like you have not choosen available options. Please try again.")
    return choice

def main():
    choice = choice_option()

    while choice != 4:
        if choice == 1:
            try:
                models = FoundationModels(bedrock_client=control_client)
                models.get_list()
            except ClientError as err:
                err_msg = err.response["Error"]["Message"]
                logger.error(f"Client Error: {err_msg}")
            else:
                logger.info("Processign Done!!!")
        elif choice == 2:  
            try:
                titan = AmazonTextModel(bedrock_client=runtime_client)
                titan.invoke_titan_model()
            except ClientError as err:
                err_msg = err.response["Error"]["Message"]
                logger.error(f"Client Error: {err_msg}")
            except BedrockException as err:
                logger.error(err.message)
            else:
                logger.info("Processign Done!!!")
        else:
            print("Looks like you have not choosen available options. Please try again.")

        choice = choice_option()
    logger.info("Thanks for using Amazon Bedrock!!!")
    exit()

main()
