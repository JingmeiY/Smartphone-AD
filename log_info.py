import logging
from logging.handlers import RotatingFileHandler


def setup_logger(name, save_log_path, max_log_size=10 * 1024 * 1024, backup_count=3):
    """
    Setup a logger for console and file logging.

    Args:
    - name (str): Name for the logger.
    - save_log_path (str): Path where the log should be saved.
    - max_log_size (int, optional): Maximum log file size before rotation. Default is 10MB.
    - backup_count (int, optional): Number of backup files to keep. Default is 3.

    Returns:
    - logger: Configured logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all messages
    logger.propagate = False  # To avoid duplicate log messages

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with log rotation
    file_handler = RotatingFileHandler(
        save_log_path, maxBytes=max_log_size, backupCount=backup_count
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger