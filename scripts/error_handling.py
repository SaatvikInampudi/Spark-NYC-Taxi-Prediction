# scripts/error_handling.py

import logging

def handle_exception(logger, e):
    """
    Logs the exception details and exits the script.
    """
    logger.error("An error occurred: ", exc_info=True)
    exit(1)