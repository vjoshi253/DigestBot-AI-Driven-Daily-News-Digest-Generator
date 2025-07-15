import logging
import requests
from bs4 import BeautifulSoup
import log_config  # Ensure logging is configured

logger = logging.getLogger(__name__)


class Scraper:
    def __init__(self, url: str):
        self.url = url

    def _fetch_html(self) -> str:
        """
        Fetches the HTML content of the page at the specified URL.

        Returns:
            str: The HTML content of the page if successful, otherwise an empty string.
        """
        try:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
            logger.info(f"Page fetched successfully: {self.url}")
            return response.text

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching the page {self.url}: {e}")
            return ""

    def _extract_text(self, html: str) -> str:
        """
        Extracts and returns the visible text content from the provided HTML string.

        This method parses the HTML, removes all <script> and <style> elements, and then extracts the visible text.
        If an error occurs during parsing, it logs the error and returns an empty string.

        Args:
            html (str): The HTML content as a string.

        Returns:
            str: The extracted visible text, or an empty string if parsing fails.
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            visible_text = soup.get_text(separator=" ", strip=True)
            logger.info(f"Page parsed successfully: {self.url}")
            return visible_text
        except Exception as e:
            logging.error(f"Error parsing HTML from {self.url}: {e}")
            return ""

    def run(self) -> str:
        """
        Runs the scraper to fetch and extract visible text from the URL.

        Returns:
            str: The extracted visible text from the page, or an empty string if fetching or parsing fails.
        """
        logger.info(f"Starting scraper for URL: {self.url}")
        html = self._fetch_html()
        if not html:
            return ""
        body_text = self._extract_text(html)
        return body_text
