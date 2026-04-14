import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_retrain_api_key() -> str:
    key = os.getenv("RETRAIN_API_KEY", "")
    if not key:
        raise ValueError("RETRAIN_API_KEY environment variable is not set.")
    return key


def verify_retrain_key(provided_key: str) -> bool:
    try:
        expected = get_retrain_api_key()
        return provided_key == expected
    except ValueError:
        return False
