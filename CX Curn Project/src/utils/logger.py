"""
A tiny, reusable logger.

WHY: `print()` is fine for a script you run once. In a real system you want
timestamps, log levels (INFO/WARNING/ERROR), and a consistent format so logs
can be shipped to a file or a monitoring system later. We centralize it so
every module logs the same way.
"""
import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid adding duplicate handlers on re-import
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
