class BedrockException(Exception):
    def __init__(self, message) -> None:
        self.message = message

class ImageException(Exception):
    def __init__(self, message):
        self.message = message
