import boto3
import json

## Creating session with AWS profile
session = boto3.Session(profile_name="bedrock-profile")

# bedrock-runtime – Contains runtime plane APIs for making inference requests for models hosted in Amazon Bedrock
bedrock_client = session.client("bedrock-runtime")

MODEL_ID_TITAN = "amazon.titan-tg1-large"
MAX_TOKEN_COUNT = 256
STOP_SEQUENCES = []
TEMPERATURE = 0.9
TOP_P = 1.0


def invoke_titan_model(question: str):
    """
    Invoke Amazon Titan Models

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

    1. maxTokenCount:
    Configures the max number of tokens to use in the generated response. (int, defaults to 512)

    2. stopSequences:
    Used to make the model stop at a desired point, such as the end of a sentence or a list.
    The returned response will not contain the stop sequence. (defaults to blank)

    --> Request Structure: Json with following propertis

    body: Input data in the format specified in the content-type request header.
    modelId: Identifier of the foundation model.
    accept:  The desired MIME type of the inference body in the response.
    contentType: The MIME type of the input data in the request

    --> Response Structure: Json with following properties

    body: Inference response from the model in the format specified in the content-type header field.
    contentType: The MIME type of the inference result.

    """

    input = json.dumps(
        dict(
            inputText=question,
            textGenerationConfig=dict(
                maxTokenCount=MAX_TOKEN_COUNT,
                stopSequences=STOP_SEQUENCES,
                temperature=TEMPERATURE,
                topP=TOP_P,
            ),
        )
    )

    ### Invoke Foundation Model
    output = bedrock_client.invoke_model(
        body=input,
        modelId=MODEL_ID_TITAN,
        accept="application/json",
        contentType="application/json",
    )

    ### Process Response
    response = json.loads(output["body"].read())
    print(response["results"][0]["outputText"])


def main():
    question = "Why do we dream?"
    invoke_titan_model(question)


main()
