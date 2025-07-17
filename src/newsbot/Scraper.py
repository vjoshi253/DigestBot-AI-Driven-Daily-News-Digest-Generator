import logging
import requests
from bs4 import BeautifulSoup
from readability import Document
from newsbot import log_config  # Ensure logging is configured

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
            doc = Document(html)
            html = doc.summary()
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            visible_text = soup.get_text(separator=" ", strip=True)
            logger.info(f"Page content parsed successfully: {self.url}")
            return visible_text
        except Exception as e:
            logging.error(f"Error parsing HTML from {self.url}: {e}")
            return ""

    def _extract_title(self, html: str) -> str:
        """
        Extracts and returns the title from the provided HTML string.

        Args:
            html (str): The HTML content as a string.

        Returns:
            str: The page title, or an empty string if not found or on error.
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""
            logger.info(f"Page title parsed successfully: {self.url}")
            return title
        except Exception as e:
            logger.error(f"Error extracting title from {self.url}: {e}")
        return ""

    def run(self) -> dict:
        """
        Runs the scraper to fetch and extract the title and visible text content from the URL.

        Returns:
            dict: A dictionary containing 'title' and 'content' keys. If fetching or parsing
            fails, values will be empty strings.
        """

        logger.info(f"Starting scraper for URL: {self.url}")
        html = self._fetch_html()
        if not html:
            return {"title": "", "content": ""}

        title = self._extract_title(html)
        content = self._extract_text(html)

        return {
            "title": title,
            "content": content,
        }
