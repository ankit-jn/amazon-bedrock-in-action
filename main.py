import boto3
from botocore.exceptions import ClientError
import logging

from utils.exception_handler import BedrockException

from list_models import FoundationModels
from model_invocation.text.amazon_titan import AmazonTitanTextGenerator
from model_invocation.text.anthropic_claude import AnthropicClaudeTextGenerator
from model_invocation.text.meta_llama2 import MetaLlama2TextGenerator
from model_invocation.text.ai21_jurassic import AI21Jurassic2TextGenerator
from model_invocation.text.cohere_command import CohereCommandTextGenerator
from model_invocation.image.stability_diffusion import StabilityDiffusionImageGenerator
from model_invocation.image.amazon_titan import AmazonTitanImageGenerator
from model_invocation.embedding.amazon_titan import AmazonTitanEmbeddeing

## Instantiate Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

## Creating session with AWS profile
session = boto3.Session(profile_name="bedrock-profile")

# bedrock – Contains runtime plane APIs for making inference requests for models hosted in Amazon Bedrock
control_client = session.client("bedrock")

# bedrock-runtime – Contains runtime plane APIs for making inference requests for models hosted in Amazon Bedrock
runtime_client = session.client("bedrock-runtime")


def test_amazon_titan_text_generator(streaming=False):
    """
    Initiator for Testing Amazon Titan Text Model
    """

    try:
        titan = AmazonTitanTextGenerator(bedrock_client=runtime_client)
        titan.process(streaming)
    except ClientError as err:
        err_msg = err.response["Error"]["Message"]
        logger.error(f"Client Error: {err_msg}")
    except BedrockException as err:
        logger.error(err.message)
    else:
        logger.info("Processign Done!!!")


def test_anthropic_claude_text_generator(streaming=False):
    """
    Initiator for Testing Anthropic Claude Text Model
    """

    try:
        claude = AnthropicClaudeTextGenerator(bedrock_client=runtime_client)
        claude.process(streaming)
    except ClientError as err:
        err_msg = err.response["Error"]["Message"]
        logger.error(f"Client Error: {err_msg}")
    except BedrockException as err:
        logger.error(err.message)
    else:
        logger.info("Processign Done!!!")


def test_meta_llama2_text_generator(streaming=False):
    """
    Initiator for Testing Meta Llama2 Text Model
    """

    try:
        llama2 = MetaLlama2TextGenerator(bedrock_client=runtime_client)
        llama2.process(streaming)
    except ClientError as err:
        err_msg = err.response["Error"]["Message"]
        logger.error(f"Client Error: {err_msg}")
    except BedrockException as err:
        logger.error(err.message)
    else:
        logger.info("Processign Done!!!")


def test_ai21_j2_text_generator():
    """
    Initiator for Testing AI21 Jurrasic 2 Text Model
    """

    try:
        j2 = AI21Jurassic2TextGenerator(bedrock_client=runtime_client)
        j2.process()
    except ClientError as err:
        err_msg = err.response["Error"]["Message"]
        logger.error(f"Client Error: {err_msg}")
    except BedrockException as err:
        logger.error(err.message)
    else:
        logger.info("Processign Done!!!")


def test_cohere_command_text_generator(streaming=False):
    """
    Initiator for Testing AI21 Jurrasic 2 Text Model
    """

    try:
        j2 = CohereCommandTextGenerator(bedrock_client=runtime_client)
        j2.process(streaming)
    except ClientError as err:
        err_msg = err.response["Error"]["Message"]
        logger.error(f"Client Error: {err_msg}")
    except BedrockException as err:
        logger.error(err.message)
    else:
        logger.info("Processign Done!!!")


def test_amazon_titan_image_generator():
    """
    Initiator for Testing Amazon Titan Image Model
    """

    try:
        titan = AmazonTitanImageGenerator(bedrock_client=runtime_client)
        titan.process()
    except ClientError as err:
        err_msg = err.response["Error"]["Message"]
        logger.error(f"Client Error: {err_msg}")
    except BedrockException as err:
        logger.error(err.message)
    else:
        logger.info("Processign Done!!!")


def test_sdxl_image_generator():
    """
    Initiator for Testing Stability Diffusion Image Model
    """

    try:
        sdxl = StabilityDiffusionImageGenerator(bedrock_client=runtime_client)
        sdxl.process()
    except ClientError as err:
        err_msg = err.response["Error"]["Message"]
        logger.error(f"Client Error: {err_msg}")
    except BedrockException as err:
        logger.error(err.message)
    else:
        logger.info("Processign Done!!!")


def test_amazon_titan_embedding():
    """
    Initiator for Testing Amazon Titan Text Model
    """

    try:
        titan = AmazonTitanEmbeddeing(bedrock_client=runtime_client)
        titan.process()
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
    print("6. Test Meta Llama2 Text Model")
    print("7. Test Meta Llama2 Text Model (with streaming)")
    print("8. Test AI21 Jurrasic 2 Text Model")
    print("9. Test Cohere Command Text Model")
    print("10. Test Cohere Command Text Model (with streaming)")
    print("11. Test Amazon Titan Image Generator Model")
    print("12. Test Stability Diffusion Image Generator Model")
    print("13. Test Amazon Titan Embedding Model")

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
            test_amazon_titan_text_generator()
        elif choice == 3:
            test_amazon_titan_text_generator(streaming=True)
        elif choice == 4:
            test_anthropic_claude_text_generator()
        elif choice == 5:
            test_anthropic_claude_text_generator(streaming=True)
        elif choice == 6:
            test_meta_llama2_text_generator()
        elif choice == 7:
            test_meta_llama2_text_generator(streaming=True)
        elif choice == 8:
            test_ai21_j2_text_generator()
        elif choice == 9:
            test_cohere_command_text_generator()
        elif choice == 10:
            test_cohere_command_text_generator(streaming=True)
        elif choice == 11:
            test_amazon_titan_image_generator()
        elif choice == 12:
            test_sdxl_image_generator()
        elif choice == 13:
            test_amazon_titan_embedding()
        else:
            print(
                "Looks like you have not choosen available options. Please try again."
            )

        choice = choice_option()
    logger.info("Thanks for using Amazon Bedrock!!!")
    exit()


main()
