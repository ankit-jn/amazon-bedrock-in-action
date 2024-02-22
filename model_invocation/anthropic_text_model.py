import json
import logging
from utils.exception_handler import BedrockException

## Instantiate Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Refer Latest documentation for Model Ids based on your use case
MODEL_ID_CLAUDE = "anthropic.claude-v2"

# Inference Parameters Default Values
TEMPERATURE = "1.0"
TOP_P = "1.0"
TOP_K = "250"
MAX_TOKENS_TO_SAMPLE = "200"
STOP_SEQUENCES = []


class AnthropicTextModel:

    def __init__(self, bedrock_client) -> None:
        self.bedrock_client = bedrock_client
    """
    --> Anthoripc text models:

    1. Claude (anthropic.claude-v2)
    Anthropic's most powerful base model, which excels at a wide range of tasks from sophisticated dialogue and creative content generation
    to detailed instruction following. The latest version features double the context window, plus improvements across reliability,
    hallucination rates, and evidence-based accuracy in long document and RAG contexts.

    2. Claude Instant:
    A faster and cheaper yet still very capable model, which can handle a range of tasks including casual dialogue,
    text analysis, summarization, and document question-answering.

    --> Inference parameters: To control the Model Responses

    1. temperature:
    The amount of randomness injected into the response. It is defaulted to 1. The range is from 0 to 1. Value closer to 0 results into analytical / multiple choice, and closer to 1 results into creative and generative tasks.

    2. top_p:
    With nucleus sampling, Anthropic Claude computes the cumulative distribution over all the options for each subsequent token in decreasing probability order and cuts it off once it reaches a particular probability specified by top_p. (defaults to 1, range 0-1).
    Either us temperature or top_p, but not both.

    3. top_k:
    It is used to remove long tail low probability responses by sampling from the top K options for each subsequent token. (defaults to 250, range 0-500)

    3. max_tokens_to_sample:
    The maximum number of tokens to generate before stopping. (defaults to 200, range 200-4096)

    4. stop_sequences:
    Sequences that will signal the model to stop generating text.

    """

    def prepare_input(self):
        self.model_id = (
            input("Please input modelId [anthropic.claude-v2]: ").strip()
            or MODEL_ID_CLAUDE
        )

        self.temperature = float(
            input("Please input temperature [0.5]: ").strip() or TEMPERATURE
        )
        self.top_p = float(input("Please input topP [1.0]: ").strip() or TOP_P)
        self.top_k = int(input("Please input topK [250]: ").strip() or TOP_K)
        self.max_tokens_to_sample = int(
            input("Please input maxTokenCount [200]: ").strip() or MAX_TOKENS_TO_SAMPLE
        )
        self.stop_sequences = (
            input("Please comma seperated input stopSequences [None]: ").strip()
            or STOP_SEQUENCES
        )
        if type(self.stop_sequences) == str:
            self.stop_sequences = self.stop_sequences.split(",")

        self.prompt = (
            input("Please input Question [Why do we dream?]: ").strip()
            or "Why do we dream?"
        )

    def process(self, streaming=False):
        """
        Invoke Anthropic Claude Model
        """

        self.prepare_input()

        ## Prepare Input for the FM invocation
        input = json.dumps(
            dict(
                prompt=f"Human: {self.prompt} \\nAssistant:",
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                stop_sequences=self.stop_sequences,
                anthropic_version="bedrock-2023-05-31",
            )
        )

        ## Invoke Foundation Model
        if not streaming:
            output = self.bedrock_client.invoke_model(
                body=input,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )

            ## Read Response
            response = json.loads(output.get("body").read())

            logger.info(f"Completion: {response.get("completion")}")
        else:
            output = self.bedrock_client.invoke_model_with_response_stream(body=input,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",)
            
            ## Read Response
            stream_response = output.get("body")
            if stream_response:
                for event in stream_response:
                    chunk = event.get("chunk")
                    if chunk:
                        logger.info(json.loads(chunk.get("bytes").decode()))