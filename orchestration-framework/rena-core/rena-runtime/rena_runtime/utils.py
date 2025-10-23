from typing import Optional
import logging

logger = logging.getLogger("rena_runtime")


def setup_logging(name: str, verbose: Optional[bool] = False):
    logging.basicConfig(
        format="[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.getLogger(name).setLevel(logging.DEBUG if verbose else logging.INFO)
    return logging.getLogger(name)
