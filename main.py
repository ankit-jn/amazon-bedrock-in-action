import boto3
import logging

from operations import Operations

## Instantiate Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

## Creating session with AWS profile
session = boto3.Session(profile_name="bedrock-profile")

# bedrock – Contains runtime plane APIs for making inference requests for models hosted in Amazon Bedrock
control_client = session.client("bedrock")

# bedrock-runtime – Contains runtime plane APIs for making inference requests for models hosted in Amazon Bedrock
runtime_client = session.client("bedrock-runtime")

operations = Operations(control_client, runtime_client)


def text_playground_menu():
    """
    Method to take input from user to run a specific test
    """

    print("------------ Select the Text Model ------------")
    print("0. Return to Main Menu")
    print("1. Test Amazon Titan Text Model")
    print("2. Test Amazon Titan Text Model (with streaming)")
    print("3. Test Anthropic Claude Text Model")
    print("4. Test Anthropic Claude Text Model (with streaming)")
    print("5. Test Meta Llama2 Text Model")
    print("6. Test Meta Llama2 Text Model (with streaming)")
    print("7. Test AI21 Jurrasic 2 Text Model")
    print("8. Test Cohere Command Text Model")
    print("9. Test Cohere Command Text Model (with streaming)")
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


def image_playground_menu():
    """
    Method to take input from user to run a specific test
    """

    print("------------ Select the Image Model ------------")
    print("0. Return to Main Menu")
    print("1. Test Amazon Titan Image Generator Model")
    print("2. Test Stability Diffusion Image Generator Model")
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


def embedding_playground_menu():
    """
    Method to take input from user to run a specific test
    """

    print("------------ Select the Embedding Model ------------")
    print("0. Return to Main Menu")
    print("1. Test Amazon Titan Embedding Model")
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


def main_menu():
    """
    Method to take input from user to run a specific test
    """

    print("------------ Gen AI Hands-on ------------")
    print("1. List all the models")
    print("2. Playground - Text")
    print("3. Playground - Image")
    print("4. Playground - Embedding")

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


def text_playground():
    choice = text_playground_menu()

    while choice != 99:
        if choice == 0:
            main_menu()
        elif choice == 1:
            operations.test_amazon_titan_text_generator()
        elif choice == 2:
            operations.test_amazon_titan_text_generator(streaming=True)
        elif choice == 3:
            operations.test_anthropic_claude_text_generator()
        elif choice == 4:
            operations.test_anthropic_claude_text_generator(streaming=True)
        elif choice == 5:
            operations.test_meta_llama2_text_generator()
        elif choice == 6:
            operations.test_meta_llama2_text_generator(streaming=True)
        elif choice == 7:
            operations.test_ai21_j2_text_generator()
        elif choice == 8:
            operations.test_cohere_command_text_generator()
        elif choice == 9:
            operations.test_cohere_command_text_generator(streaming=True)
        else:
            print(
                "Looks like you have not choosen available options. Please try again."
            )

        choice = text_playground_menu()
    logger.info("Thanks for using Amazon Bedrock!!!")
    exit()


def image_playground():
    choice = image_playground_menu()

    while choice != 99:
        if choice == 0:
            main_menu()
        elif choice == 1:
            operations.test_amazon_titan_image_generator()
        elif choice == 2:
            operations.test_sdxl_image_generator()
        else:
            print(
                "Looks like you have not choosen available options. Please try again."
            )

        choice = image_playground_menu()
    logger.info("Thanks for using Amazon Bedrock!!!")
    exit()


def embedding_playground():
    choice = embedding_playground_menu()

    while choice != 99:
        if choice == 0:
            main_menu()
        elif choice == 1:
            operations.test_amazon_titan_embedding()
        else:
            print(
                "Looks like you have not choosen available options. Please try again."
            )

        choice = embedding_playground_menu()
    logger.info("Thanks for using Amazon Bedrock!!!")
    exit()


def main():
    choice = main_menu()

    while choice != 99:
        if choice == 1:
            operations.list_models()
        elif choice == 2:
            text_playground()
        elif choice == 3:
            image_playground()
        elif choice == 4:
            embedding_playground()
        else:
            print(
                "Looks like you have not choosen available options. Please try again."
            )

        choice = main_menu()

    logger.info("Thanks for using Amazon Bedrock!!!")
    exit()


main()
