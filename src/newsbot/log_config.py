import logging

"""
This module configures the logging settings for the application.

Usage:
    Import this module at the start of your application to enable logging as configured.
    After importing, obtain a logger in your modules using:
        logger = logging.getLogger(__name__)
    The logger will inherit the configuration set up by this module.
"""
LOG_FILE = "NewsLetter.log"

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",  # or "w" to overwrite
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
