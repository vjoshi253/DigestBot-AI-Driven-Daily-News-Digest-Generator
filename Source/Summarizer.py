import logging
from Scraper import Scraper
from transformers import pipeline
from device_config import DeviceManager
import log_config  # Ensure logging is configured


logger = logging.getLogger(__name__)


class Summarizer:
    """
    A class for extracting and summarizing text content from a given URL using a Hugging Face summarization model.

    Attributes:
        url (str): The URL of the page to summarize.
        device (torch.device or str): The device identifier for running the model (e.g., CPU or GPU).
        model: The Hugging Face summarization pipeline.

    Args:
        url (str): The URL of the page to summarize.
        model_name (str, optional): The name of the Hugging Face summarization model to use. Defaults to "facebook/bart-large-cnn".

    """

    def __init__(self, url: str, model_name: str = "facebook/bart-large-cnn"):
        self.url = url
        self.device = DeviceManager.get_torch_type()
        self.model = pipeline(
            "summarization",
            model=model_name,
            device=self.device,
        )

    def _summarize(self, text: str, max_length: int, min_length: int) -> str:
        """
        Generates a summary of the provided text using the loaded model.

        Args:
            text (str): The input text to be summarized.
            max_length (int): The maximum length of the generated summary.
            min_length (int): The minimum length of the generated summary.

        Returns:
            str: The summarized text if successful, otherwise an empty string.
        """
        try:
            summary = self.model(
                text, max_length=max_length, min_length=min_length, do_sample=False
            )
            logger.info(f"Page summarized successfully: {self.url}")
            return summary[0]["summary_text"]
        except Exception as e:
            logger.error(
                f"Error during summarization {self.url}: {type(e).__name__}: {e}"
            )
            return ""

    def _get_summary_lengths(
        self, text: str, max_cap: int = 250, min_floor: int = 30, ratio: float = 0.4
    ) -> tuple:
        """
        Calculates dynamic maximum and minimum summary lengths based on the input text and provided constraints.

        Args:
            text (str): The input text to be summarized.
            max_cap (int, optional): The upper limit for the maximum summary length. Defaults to 150.
            min_floor (int, optional): The lower limit for the minimum summary length. Defaults to 30.
            ratio (float, optional): The ratio of the input text's word count to use for the maximum summary length.
            Defaults to 0.3.

        Returns:
            tuple: A tuple (max_length, min_length) where:
                - max_length (int): The calculated maximum summary length, constrained by `max_cap` and `ratio`.
                - min_length (int): The calculated minimum summary length, constrained by `min_floor` and `max_length`.
        """
        word_count = len(text.split())
        dynamic_max = int(word_count * ratio)

        max_length = min(dynamic_max, max_cap)
        min_length = (
            min(min_floor, max_length - 1)
            if max_length > min_floor
            else int(0.6 * max_length)
        )

        return max_length, min_length

    def run(self, content: str) -> str:
        """
        Generates a summary of the provided text content.

        Args:
            content (str): The content to be summarized.

        Returns:
            str: The summarized text if content is found; otherwise, an empty string.
        """
        max_length, min_length = self._get_summary_lengths(content)
        return self._summarize(content, max_length, min_length)
