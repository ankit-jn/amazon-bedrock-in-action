import boto3
from botocore.exceptions import ClientError
import logging

from utils.exception_handler import BedrockException

from list_models import FoundationModels
from model_invocation.amazon_text_model import AmazonTextModel
from model_invocation.anthropic_text_model import AnthropicTextModel

## Instantiate Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

## Creating session with AWS profile
session = boto3.Session(profile_name="bedrock-profile")

# bedrock – Contains runtime plane APIs for making inference requests for models hosted in Amazon Bedrock
control_client = session.client("bedrock")

# bedrock-runtime – Contains runtime plane APIs for making inference requests for models hosted in Amazon Bedrock
runtime_client = session.client("bedrock-runtime")


def test_amazon_titan(streaming=False):
    """
    Initiator for Testing Amazon Titan Text Model
    """

    try:
        titan = AmazonTextModel(bedrock_client=runtime_client)
        titan.process(streaming)
    except ClientError as err:
        err_msg = err.response["Error"]["Message"]
        logger.error(f"Client Error: {err_msg}")
    except BedrockException as err:
        logger.error(err.message)
    else:
        logger.info("Processign Done!!!")

def test_anthropic_claude(streaming=False):
    """
    Initiator for Testing Anthropic Claude Text Model
    """

    try:
        claude = AnthropicTextModel(bedrock_client=runtime_client)
        claude.process(streaming)
    except ClientError as err:
        err_msg = err.response["Error"]["Message"]
        logger.error(f"Client Error: {err_msg}")
    except BedrockException as err:
        logger.error(err.message)
    else:
        logger.info("Processign Done!!!")


def list_models():
    """
    Initiator for listing FM models deployed with Amazon Bedrock
    """

    try:
        models = FoundationModels(bedrock_client=control_client)
        models.get_list()
    except ClientError as err:
        err_msg = err.response["Error"]["Message"]
        logger.error(f"Client Error: {err_msg}")
    else:
        logger.info("Processign Done!!!")


def choice_option():
    """
    Method to take input from user to run a specific test
    """

    print("------------ Gen AI Hands-on ------------")
    print("1. List all the models")
    print("2. Test Amazon Titan Text Model")
    print("3. Test Amazon Titan Text Model (with streaming)")
    print("4. Test Anthropic Claude Text Model")
    print("5. Test Anthropic Claude Text Model (with streaming)")
    print("99. Exit")
    valid = False
    while not valid:
        choice = input("Please select option: ").strip()
        if choice.isnumeric():
            valid = True
            choice = int(choice)
        else:
            print(
                "Looks like you have not choosen available options. Please try again."
            )
    return choice


def main():
    choice = choice_option()

    while choice != 99:
        if choice == 1:
            list_models()
        elif choice == 2:
            test_amazon_titan()
        elif choice == 3:
            test_amazon_titan(streaming=True)
        elif choice == 4:
            test_anthropic_claude()
        elif choice == 5:
            test_anthropic_claude(streaming=True)
        else:
            print(
                "Looks like you have not choosen available options. Please try again."
            )

        choice = choice_option()
    logger.info("Thanks for using Amazon Bedrock!!!")
    exit()


main()
