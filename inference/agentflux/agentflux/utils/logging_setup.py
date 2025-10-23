# logging_setup.py
import logging
import sys

def setup_logging(log_file_path: str):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))

    # 文件输出
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Suppress noisy third-party INFO logs (e.g., httpx request lines)
    for noisy_logger in ("httpx", "httpcore"):
        nl = logging.getLogger(noisy_logger)
        nl.setLevel(logging.WARNING)
        # Optionally prevent propagation if upstream handlers are very permissive
        # nl.propagate = False
