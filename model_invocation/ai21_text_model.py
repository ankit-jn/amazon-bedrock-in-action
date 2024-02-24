import json
import logging

## Instantiate Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Refer Latest documentation for Model Ids based on your use case
MODEL_ID_J2 = "ai21.j2-ultra"

# Inference Parameters Default Values
TEMPERATURE = "0.7"
TOP_P = "1.0"
MAX_TOKENS = "200"
STOP_SEQUENCES = []

PRESENCE_PENALTY = 0
COUNT_PENALTY = 0
FREQUENCY_PENALTY = 0


class AI21textModel:
    """
    --> AI21 text models:

    1. ai21.j2-ultra
    2. ai21.j2-mid
    1. ai21.j2-jumbo-instruct
    2. ai21.j2-grande-instruct

    --> Inference parameters: To control the Model Responses

    A. Randomness and Diversity ###

        1. temperature:
        Modulates the probability density function for the next tokens, implementing the temperature sampling technique.
        This parameter can be used to deepen or flatten the density function curve.
        A lower value results in a steeper curve and more deterministic responses,
        whereas a higher value results in a flatter curve and more random (creative and different) responses for the same prompt.
        (float, defaults to 0, max value is 1.5)

        2. topP:
        Top P controls token choices, based on the probability of the potential choices. If you set Top P below 1.0,
        the model considers only the most probable options and ignores less probable options.
        The result is more stable and repetitive completions. (defaults to 0.9, max value is 1.0)

    B. Length

        1. maxTokens:
        Configures the max number of tokens to use in the generated response. (int, defaults to 200)

        2. stopSequences:
        Used to make the model stop at a desired point, such as the end of a sentence or a list.
        The returned response will not contain the stop sequence. (defaults to blank)

    C. Repitions
        1. Presence penalty:
        Reduce the probability of generating new tokens that appeared at least once in the prompt or in the completion.

        2. Count penalty:
        Reduce the probability of generating new tokens that appeared in the prompt or in the completion, proportional
        to the number of appearances.

        3. Frequency penalty:
        Reduce the probability of generating new tokens that appeared in the prompt or in the completion, proportional
        to the frequency of their appearances in the text (normalized to text length).

        4. Penalize special tokens:
        bedrock.helpPanel.ai21Repetitions.listPenalizeSpecialTokens (Whitespaces, Numbers, Emojis, Punctuations, Stopwords)


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
            "topP": float,
            "maxTokens": int,
            "stopSequences": [string],
            "countPenalty": {
                "scale": float
            },
            "presencePenalty": {
                "scale": float
            },
            "frequencyPenalty": {
                "scale": float
            }
        }

        "countPenalty": {
            "scale": float,
            "applyToWhitespaces": boolean,
            "applyToPunctuations": boolean,
            "applyToNumbers": boolean,
            "applyToStopwords": boolean,
            "applyToEmojis": boolean
        }

    --> Response Structure: Json with following properties

        ## Plain Response
        {
            'id': int,
            'prompt': []
            'completions': []
        }

    """

    def __init__(self, bedrock_client) -> None:
        self.bedrock_client = bedrock_client

    def prepare_input(self):
        self.model_id = (
            input("Please input modelId [ai21.j2-ultra]: ").strip() or MODEL_ID_J2
        )

        self.temperature = float(
            input("Please input temperature [0.7]: ").strip() or TEMPERATURE
        )
        self.top_p = float(input("Please input topP [1.0]: ").strip() or TOP_P)
        self.maxTokens = int(
            input("Please input maxTokens [512]: ").strip() or MAX_TOKENS
        )
        self.stop_sequences = (
            input("Please comma seperated input stopSequences [None]: ").strip()
            or STOP_SEQUENCES
        )
        if type(self.stop_sequences) == str:
            self.stop_sequences = self.stop_sequences.split(",")

        self.presence_penalty = float(
            input("Please input presence_penalty [0]: ").strip() or PRESENCE_PENALTY
        )
        self.count_penalty = float(
            input("Please input count_penalty [0]: ").strip() or COUNT_PENALTY
        )
        self.frequency_penalty = float(
            input("Please input frequency_penalty [0]: ").strip() or FREQUENCY_PENALTY
        )

        self.prompt = (
            input("Please input Question [Why do we dream?]: ").strip()
            or "Why do we dream?"
        )

    def process(self):
        """
        Invoke AI21 Text Model
        """
        ## Collect user Inputs
        self.prepare_input()

        ### Prepare Input for the FM invocation
        input = json.dumps(
            dict(
                prompt=self.prompt,
                maxTokens=self.maxTokens,
                temperature=self.temperature,
                topP=self.top_p,
                stopSequences=self.stop_sequences,
                countPenalty=dict(scale=self.count_penalty),
                presencePenalty=dict(scale=self.presence_penalty),
                frequencyPenalty=dict(scale=self.frequency_penalty),
            )
        )

        output = self.bedrock_client.invoke_model(
            body=input,
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json",
        )

        ### Read Response
        response = json.loads(output["body"].read())
        
        for result in response["completions"]:
            logger.info(f"Output text: {result["data"]["text"]}")
