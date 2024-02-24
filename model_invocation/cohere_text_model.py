import json
import logging
from utils.exception_handler import BedrockException

## Instantiate Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Refer Latest documentation for Model Ids based on your use case
MODEL_ID_COMMAND = "cohere.command-text-v14"

# Inference Parameters Default Values
TEMPERATURE = "0.75"
TOP_P = "0.01"
TOP_K = "0"
MAX_TOKENS = "400"
STOP_SEQUENCES = []
RETURN_LIKELIHOODS = "NONE"

class CohereTextModel:
    """
    --> Amazon text models:

    1. cohere.command-light-text-v14:
    2. cohere.command-text-v14: 
    

    --> Inference parameters: To control the Model Responses

    A. Randomness and Diversity ###

        1. temperature:
        A non-negative float that tunes the degree of randomness in generation. Lower temperatures mean less random generations.
        (defaults to 0.75, range: 0-5.0)

        2. p:
        bedrock.helpPanel.cohereRandomDiversity.listP (defaults to 0.01, range: 0.01-0.99)

        2. k
        Ensures only the top k most likely tokens are considered for generation at each step. (defaults to 0, range: 0-500)

    B. Length

        1. max_tokens:
        Configures the max number of tokens to use in the generated response. (defaults to 400, range: 0-4000)

        2. stop_sequences:
        Used to make the model stop at a desired point, such as the end of a sentence or a list.
        The returned response will not contain the stop sequence. (defaults to blank)

        2. return_likelihoods:
        Used to make the model stop at a desired point, such as the end of a sentence or a list.
        The returned response will not contain the stop sequence. (defaults to blank)

        
    --> Request Structure: Json with following propertis

        {
            "body": string          ## Input data in the format specified in the content-type request header.
            "modelId": string       ## Identifier of the foundation model.
            "accept":  string       ## The desired MIME type of the inference body in the response.
            "contentType": string   ## The MIME type of the input data in the request
        }

        Request Body
        {
            "prompt": string,
            "temperature": float,
            "p": float,
            "k": float,
            "max_tokens": int,
            "stop_sequences": [string],
            "return_likelihoods": "GENERATION|ALL|NONE",
            "stream": boolean,
            "num_generations": int,
            "logit_bias": {token_id: bias},
            "truncate": "NONE|START|END"
        }

    --> Response Structure: Json with following properties

        ## Plain Response
        {
            "generations": [
                {
                    "finish_reason": "COMPLETE | MAX_TOKENS | ERROR | ERROR_TOXIC",
                    "id": string,
                    "text": string,
                    "likelihood" : float,
                    "token_likelihoods" : [{"token" : float}],
                    "is_finished" : true | false,
                    "index" : integer
                
                }
            ],
            "id": string,
            "prompt": string
        }

        ## Stream Response
        {
            'chunk': {
                'bytes': b'{
                    "index": int,
                    "is_finished": Boolean,
                    "text": <response-chunk>
                }'
            }
        }

    """
    def __init__(self, bedrock_client) -> None:
        self.bedrock_client = bedrock_client

    def prepare_input(self):
        self.model_id = input('Please input modelId [cohere.command-text-v14]: ').strip() or MODEL_ID_COMMAND

        self.temperature = float(input('Please input temperature [0.75]: ').strip() or TEMPERATURE)
        self.top_p = float(input('Please input p [0.01]: ').strip() or TOP_P)
        self.top_k = int(input('Please input k [0]: ').strip() or TOP_K)
        self.max_tokens = int(input('Please input maxTokenCount [400]: ').strip() or MAX_TOKENS)
        self.stop_sequences = (input('Please comma seperated input stopSequences [None]: ').strip() or STOP_SEQUENCES)
        if type(self.stop_sequences) == str:
            self.stop_sequences = self.stop_sequences.split(",")
        self.return_likelihoods = input('Please input return_likelihoods [NONE]: ').strip() or RETURN_LIKELIHOODS
        
        self.prompt = input('Please input Question [Why do we dream?]: ').strip() or "Why do we dream?"

    def process(self, streaming = False):
        """
        Invoke Amazon Titan Text Model
        """
        ## Collect user Inputs
        self.prepare_input()

        ### Prepare Input for the FM invocation
        input = json.dumps(
            dict(
                prompt=self.prompt,
                temperature=self.temperature,
                p=self.top_p,
                k=self.top_k,
                max_tokens=self.max_tokens,
                stop_sequences=self.stop_sequences,
                return_likelihoods= self.return_likelihoods,
                num_generations= 2,
                stream=streaming,
            )
        )

        if not streaming:
            ### Invoke Foundation Model
            output = self.bedrock_client.invoke_model(
                body=input,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )

            ### Read Response
            response = json.loads(output["body"].read())
            
            for result in response["generations"]:
                logger.info(result["text"])
                logger.info(f"Finish Reason: {result["finish_reason"]}")
                if 'likelihood' in result:
                    logger.info(f"Likelihood: {result['likelihood']}\n")
        else:
            output = self.bedrock_client.invoke_model_with_response_stream(
                body=input,
                modelId=self.model_id,
                accept = "application/json",
                contentType="application/json"
            )

            response_stream = output.get("body")
            ## Process Stream
            for event in response_stream:
                chunk = event.get("chunk")
                data = json.loads(chunk.get("bytes").decode())
                text = data.get("text")
                if '\n' == text:
                    print('')
                    continue
                print(text, end='')
