import json
import logging

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


class AnthropicClaudeTextGenerator:

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
    The amount of randomness injected into the response. It is defaulted to 1. The range is from 0 to 1. 
    Value closer to 0 results into analytical / multiple choice, and closer to 1 results into creative and generative tasks.

    2. top_p:
    With nucleus sampling, Anthropic Claude computes the cumulative distribution over all the options for each subsequent token 
    in decreasing probability order and cuts it off once it reaches a particular probability specified by top_p. (defaults to 1, range 0-1).
    Either us temperature or top_p, but not both.

    3. top_k:
    It is used to remove long tail low probability responses by sampling from the top K options for each subsequent token. (defaults to 250, range 0-500)

    3. max_tokens_to_sample:
    The maximum number of tokens to generate before stopping. (defaults to 200, range 200-4096)

    4. stop_sequences:
    Sequences that will signal the model to stop generating text.

    
    --> Request Structure: Json with following propertis
        {
            "body": string          ## Input data in the format specified in the content-type request header.
            "modelId": string       ## Identifier of the foundation model.
            "accept":  string       ## The desired MIME type of the inference body in the response.
            "contentType": string   ## The MIME type of the input data in the request
        }

        Request Body
        {
            "prompt": "\n\nHuman:<prompt>\n\nAssistant:",
            "temperature": float,
            "top_p": float,
            "top_k": int,
            "max_tokens_to_sample": int,
            "stop_sequences": [string]
        }

    --> Response Structure: Json with following propertis
        
        ## Plain Response
        {
            "completion": string,   ## The resulting completion up to and excluding the stop sequences.
            "stop_reason": string,  ## The reason why the model stopped generating the response. Values: stop_sequence, max_tokens
            "stop": string          ## contains the stop sequence that signalled the model to stop generating text.
        }

    """

    def prepare_input(self):
        self.model_id = (
            input("Please input modelId [anthropic.claude-v2]: ").strip()
            or MODEL_ID_CLAUDE
        )

        self.temperature = float(
            input("Please input temperature [0.5]: ").strip() or TEMPERATURE
        )
        self.top_p = float(input("Please input top_p [1.0]: ").strip() or TOP_P)
        self.top_k = int(input("Please input top_k [250]: ").strip() or TOP_K)
        self.max_tokens_to_sample = int(
            input("Please input max_tokens_to_sample [200]: ").strip() or MAX_TOKENS_TO_SAMPLE
        )
        self.stop_sequences = (
            input("Please comma seperated input stop_sequences [None]: ").strip()
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
                max_tokens_to_sample=self.max_tokens_to_sample,
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
            response_stream = output.get("body")

            ## Process Stream
            for event in response_stream:
                chunk = event.get("chunk")
                data = json.loads(chunk.get("bytes").decode())
                logger.info(f"Completion: {data["cmpletion"]}")
