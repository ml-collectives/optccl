import logging
import sys

ROOT_LOGGER_NAME = "optccl"


def configure_logging(
    verbosity: int = 0, quiet: bool = False, log_file: str | None = None
) -> None:
    if quiet:
        level = logging.WARNING
    elif verbosity >= 1:
        level = logging.DEBUG
    else:
        level = logging.INFO

    fmt = "[%(levelname)s] %(name)s: %(message)s" if verbosity >= 1 else "%(message)s"
    formatter = logging.Formatter(fmt)

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
