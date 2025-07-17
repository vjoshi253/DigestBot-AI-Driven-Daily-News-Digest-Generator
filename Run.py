import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from newsbot.Scraper import Scraper
from newsbot.Summarizer import Summarizer
from newsbot.ArticleEvaluator import ArticleEvaluator

if __name__ == "__main__":
    url = "https://www.utsa.edu/today/2025/07/story/AI-for-everyone-camp.html"
    scraper = Scraper(url)
    page = scraper.run()

    if page["content"]:
        print("Title:", page["title"])
        summarizer = Summarizer(url)
        summary = summarizer.run(page["content"])
        if summary:
            print("Summary:", summary)

        ArticleEvaluator = ArticleEvaluator()
        rating = ArticleEvaluator.evaluate(
            title=page["title"],
            content=page["content"],
            interest="Artificial Intelligence",
            user_type="Power User",
        )
        if rating is not None:
            print("Rating:", rating)
