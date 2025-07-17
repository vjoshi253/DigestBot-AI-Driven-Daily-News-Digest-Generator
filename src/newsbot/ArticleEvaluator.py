import re
import logging
from typing import Union
from transformers import pipeline
from newsbot.device_config import DeviceManager
from newsbot import log_config  # Ensure logging is configured


logger = logging.getLogger(__name__)


class ArticleEvaluator:
    """
    ArticleEvaluator is a class designed to evaluate news articles using a language model,
    providing a numerical rating based on user-specific criteria.

    Attributes:
        model_name (str): The name or path of the language model to use for evaluation.
        model: The instantiated text-generation pipeline for generating article ratings.
    """

    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM3-3B"):
        self.model_name = model_name
        self.model = pipeline(
            "text-generation",
            model=self.model_name,
            device=DeviceManager.get_torch_device(),
        )

    def _build_prompt(
        self, title: str, content: str, interest: str, user_type: str
    ) -> str:
        """
        Constructs a prompt string for evaluating a news article based on user-specific criteria.

        Args:
            title (str): The title of the article.
            content (str): The full content of the article.
            interest (str): The user's area of interest.
            user_type (str): The type of user ("Power User" or "Basic User").

        Returns:
            str: A formatted prompt instructing a model to rate the article on a scale of 1 to 10,
            considering clarity, relevance, user interest, and expertise level, with specific
            guidelines for different user types.
        """
        return (
            "You are a senior news editor evaluating an article for a specific reader. \n"
            "Rate the article strictly on a scale of 1 to 10 based on the following criteria:\n\n"
            "- Clarity and relevance of the article\n"
            "- The user's interest area\n"
            "- The user's expertise level (Power User or Basic User)\n\n"
            "Do NOT provide any explanation. Only output the rating as a single number (e.g., 7).\n\n"
            "Guidelines:\n"
            "- For a Power User, prioritize articles that are technical, in-depth, and impactful.\n"
            "- For a Basic User, prioritize articles that are simple, clear, and essential for general awareness.\n\n"
            f"Title: {title.strip()}\n\n"
            f"Article Content: {content.strip()}\n\n"
            f"User Interest: {interest.strip()}\n\n"
            f"User Type: {user_type.strip()}\n"
        )

    def evaluate(
        self, title: str, content: str, interest: str, user_type: str
    ) -> Union[int, None]:
        """
        Evaluates an article based on its title, content, user interest, and user type.

        Args:
            title (str): The title of the article.
            content (str): The content of the article.
            interest (str): The area of interest or topic for evaluation.
            user_type (str): The type of user for whom the evaluation is being performed.

        Returns:
            Union[int, None]: An integer score between 1 and 10 if evaluation is successful, otherwise None.
        """
        logger.info(f"Starting article evaluation for: '{title.strip()}'")
        prompt = self._build_prompt(title, content, interest, user_type)
        try:
            output = self.model(prompt, max_length=10, do_sample=False)
            if isinstance(output, list) and len(output) > 0:
                generated_text = output[0].get("generated_text") or output[0].get(
                    "text"
                )
                if generated_text:
                    response = generated_text.strip()
                    match = re.search(r"\b(10|[1-9])\b", response)
                    if match:
                        return int(match.group(1))
        except Exception as e:
            logger.error(f"Error evaluating article {title}: {e}", exc_info=True)
        return None
