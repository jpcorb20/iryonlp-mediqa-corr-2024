import re

from tenacity import RetryCallState
from tenacity.wait import wait_base
import numpy as np


class wait_log_extract_seconds(wait_base):
    def __init__(
        self,
        default_delay: float = 1.0,
        pattern: str = r'Try again in (\d+) seconds\.',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.default_delay = default_delay
        self.pattern = pattern

    def _get_error_msg(self, retry_state: RetryCallState) -> str:
        # Trick to catch error message.
        try:
            retry_state.outcome.result() # raise error.
        except Exception as e:
            text = str(e) # Once raised, catch error string only.
        return text

    def _compute_output(self, time: float, truncate: float = 3.0) -> float:
        offset = np.random.normal(scale=1.0)
        offset = np.clip(offset, -truncate, truncate)
        return time + float(offset)

    def __call__(self, retry_state: RetryCallState) -> float:
        error_msg = self._get_error_msg(retry_state)
        match = re.search(self.pattern, error_msg)
        if match:
            time = float(match.group(1))
            return self._compute_output(time)
        else:
            return self.default_delay
