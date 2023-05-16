import logging
import os
import sys

LOGS_DIR = str(os.getenv("LOGS_DIR"))
LOGGER_LEVEL = str(os.getenv("LOGGER_LEVEL"))

# Configure Logging
FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(process)d - "
    "%(levelname)s - %(message)s - %(module)s - "
    "%(funcName)s"
)


if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "logs.log")


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf8")
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(LOGGER_LEVEL)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    logger.propagate = False
    return logger
