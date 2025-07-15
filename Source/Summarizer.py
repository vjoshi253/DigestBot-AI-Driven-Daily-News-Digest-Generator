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

    def run(self, max_length=1000, min_length=500) -> str:
        """
        Generates a summary of the text content retrieved from the specified URL.

        Args:
            max_length (int, optional): The maximum length of the summary. Defaults to 1000.
            min_length (int, optional): The minimum length of the summary. Defaults to 500.

        Returns:
            str: The summarized text if content is found; otherwise, an empty string.
        """
        scraper = Scraper(self.url)
        text = scraper.run()
        if not text:
            return ""
        return self._summarize(text, max_length, min_length)


if __name__ == "__main__":
    url = "https://www.utsa.edu/today/2025/07/story/AI-for-everyone-camp.html"
    summarizer = Summarizer(url)
    summary = summarizer.run()
    if summary:
        print("Summary:")
        print(summary)
    else:
        print("Failed to generate summary.")
