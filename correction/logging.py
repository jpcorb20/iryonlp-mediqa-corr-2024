import sys
import logging

from tenacity import RetryCallState

LOG_NAME = "semantic_kernel_engine"


def prepare_log(log_name: str = LOG_NAME, level: int = logging.DEBUG) -> logging.Logger:
    log = logging.getLogger(log_name)
    log.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(handler)
    return log


def log_attempt_number(retry_state: RetryCallState):
    """return the result of the last call attempt"""
    log = logging.getLogger(LOG_NAME)
    if retry_state.attempt_number < 1:
        loglevel = logging.INFO
    else:
        loglevel = logging.WARNING
    try:
        retry_state.outcome.result()
    except Exception as e:
        error = e
    log.log(loglevel, f"Retrying {retry_state.fn.__name__}: {retry_state.attempt_number} - {error}.")
