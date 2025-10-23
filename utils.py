import torch
import os
from huggingface_hub import login as hf_login
from dotenv import load_dotenv

# =======================[ Classes ]=======================
class CustomPrinter:
    """
    A custom printer class for consistent logging.
    """
    def __init__(self, print_fc = None, prefix: str = "-> "):
        if print_fc is None:
            self.print_fc = print
        self.prefix = prefix
    
    def set_print_fc(self, print_fc):
        """
        Set a custom print function.
        Args:
            print_fc: The custom print function to use.
        Returns:
            None
        """
        self.print_fc = print_fc

    def __call__(self, message: str, end="\n", flush=False, no_prefix=False):
        """
        Print a message with a consistent prefix.
        Args:
            message (str): The message to print.
        Returns:
            None
        """
        if no_prefix:
            prefix = ""
        else:
            prefix = self.prefix
            if message.startswith("\n"):
                prefix = "\n" + prefix
                message = message[1:]
            
        if self.print_fc == print:
            self.print_fc(f"{prefix}{message}", end=end, flush=flush)
        else:
            self.print_fc(f"{prefix}{message}")

# =======================[ Functions ]=======================
def get_device():
    """
    Determine the target device for model loading.
    Returns:
        device (str): The device to load the model onto (e.g., "cpu", "cuda", "xpu").
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.xpu.is_available():
        # Check if Intel XPU (GPU) is available
        device = "xpu"
    else:
        device =  "cpu"
    return device


def format_docs(docs):
    """
    Helper function to format a list of Document objects into a single string.
    Args:
        docs (List[Document]): List of Document objects.
    Returns:
        str: Concatenated string of document contents.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def run_hugging_face_auth(hf_env_var: str = "HUGGINGFACE_API_KEY"):
    # Load environment variables from .env file
    load_dotenv()

    # Get Hugging Face token from environment variable
    hf_token=os.getenv(hf_env_var)

    # Login to Hugging Face. Hugging Faces should automatically show a login prompt if needed.
    hf_login(token=hf_token)