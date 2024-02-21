import boto3
import json

def main():
    ## Creating session with AWS profile
    session = boto3.Session(profile_name="bedrock-profile")

    ## bedrock – Contains control plane APIs for managing, training, and deploying models. 
    bedrock_client = session.client("bedrock")

    ## List all the foundation models deployed with Amazon Bedrock
    model_list = bedrock_client.list_foundation_models()

    ## Count Models
    counts = len(model_list["modelSummaries"])
    print(f"Total Models: {counts}")

    ## Print Model Details
    print(json.dumps(model_list, indent=2))

main()