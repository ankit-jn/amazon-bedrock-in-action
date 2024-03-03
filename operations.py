import logging

from botocore.exceptions import ClientError
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
from model_invocation.embedding.cohere import CohereEmbeddeing

## Instantiate Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


class Operations:

    def __init__(self, control_client, runtime_client) -> None:
        self.control_client = control_client
        self.runtime_client = runtime_client

    def list_models(self):
        """
        Initiator for listing FM models deployed with Amazon Bedrock
        """

        try:
            models = FoundationModels(bedrock_client=self.control_client)
            models.get_list()
        except ClientError as err:
            err_msg = err.response["Error"]["Message"]
            logger.error(f"Client Error: {err_msg}")
        else:
            logger.info("Processign Done!!!")

    def generate_text_using_amazon_titan(self, streaming=False):
        """
        Initiator for Testing Amazon Titan Text Model
        """

        try:
            titan = AmazonTitanTextGenerator(bedrock_client=self.runtime_client)
            titan.process(streaming)
        except ClientError as err:
            err_msg = err.response["Error"]["Message"]
            logger.error(f"Client Error: {err_msg}")
        except BedrockException as err:
            logger.error(err.message)
        else:
            logger.info("Processign Done!!!")

    def generate_text_using_anthropic_claude(self, streaming=False):
        """
        Initiator for Testing Anthropic Claude Text Model
        """

        try:
            claude = AnthropicClaudeTextGenerator(bedrock_client=self.runtime_client)
            claude.process(streaming)
        except ClientError as err:
            err_msg = err.response["Error"]["Message"]
            logger.error(f"Client Error: {err_msg}")
        except BedrockException as err:
            logger.error(err.message)
        else:
            logger.info("Processign Done!!!")

    def generate_text_using_meta_llama2(self, streaming=False):
        """
        Initiator for Testing Meta Llama2 Text Model
        """

        try:
            llama2 = MetaLlama2TextGenerator(bedrock_client=self.runtime_client)
            llama2.process(streaming)
        except ClientError as err:
            err_msg = err.response["Error"]["Message"]
            logger.error(f"Client Error: {err_msg}")
        except BedrockException as err:
            logger.error(err.message)
        else:
            logger.info("Processign Done!!!")

    def generate_text_using_ai21_jurassic2(self):
        """
        Initiator for Testing AI21 Jurrasic 2 Text Model
        """

        try:
            j2 = AI21Jurassic2TextGenerator(bedrock_client=self.runtime_client)
            j2.process()
        except ClientError as err:
            err_msg = err.response["Error"]["Message"]
            logger.error(f"Client Error: {err_msg}")
        except BedrockException as err:
            logger.error(err.message)
        else:
            logger.info("Processign Done!!!")

    def generate_text_using_cohere_command(self, streaming=False):
        """
        Initiator for Testing AI21 Jurrasic 2 Text Model
        """

        try:
            j2 = CohereCommandTextGenerator(bedrock_client=self.runtime_client)
            j2.process(streaming)
        except ClientError as err:
            err_msg = err.response["Error"]["Message"]
            logger.error(f"Client Error: {err_msg}")
        except BedrockException as err:
            logger.error(err.message)
        else:
            logger.info("Processign Done!!!")

    def generate_image_using_amazon_titan(self):
        """
        Initiator for Testing Amazon Titan Image Model
        """

        try:
            titan = AmazonTitanImageGenerator(bedrock_client=self.runtime_client)
            titan.process()
        except ClientError as err:
            err_msg = err.response["Error"]["Message"]
            logger.error(f"Client Error: {err_msg}")
        except BedrockException as err:
            logger.error(err.message)
        else:
            logger.info("Processign Done!!!")

    def generate_image_using_sdxl(self):
        """
        Initiator for Testing Stability Diffusion Image Model
        """

        try:
            sdxl = StabilityDiffusionImageGenerator(bedrock_client=self.runtime_client)
            sdxl.process()
        except ClientError as err:
            err_msg = err.response["Error"]["Message"]
            logger.error(f"Client Error: {err_msg}")
        except BedrockException as err:
            logger.error(err.message)
        else:
            logger.info("Processign Done!!!")

    def generate_embedding_using_amazon_titan(self):
        """
        Initiator for Testing Amazon Titan Embedding
        """

        try:
            titan = AmazonTitanEmbeddeing(bedrock_client=self.runtime_client)
            titan.process()
        except ClientError as err:
            err_msg = err.response["Error"]["Message"]
            logger.error(f"Client Error: {err_msg}")
        except BedrockException as err:
            logger.error(err.message)
        else:
            logger.info("Processign Done!!!")

    def generate_embedding_using_cohere(self):
        """
        Initiator for Testing Cohere Embedding
        """

        try:
            cohere = CohereEmbeddeing(bedrock_client=self.runtime_client)
            cohere.process()
        except ClientError as err:
            err_msg = err.response["Error"]["Message"]
            logger.error(f"Client Error: {err_msg}")
        except BedrockException as err:
            logger.error(err.message)
        else:
            logger.info("Processign Done!!!")
