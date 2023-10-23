from typing import Optional, Dict, Any
import logging

import openai
logger = logging.getLogger(__name__)


class OpenAiLibWrapper:

    api_key: Optional[str] = None
    timeout: 0

    @classmethod
    def set_api_key(cls, api_key: str) -> None:
        if api_key is None:
            raise ValueError("api_key cannot be None.")
        cls.api_key = api_key
        # initialize open ai library
        openai.api_key = api_key

    @classmethod
    def get_api_key(cls) -> Optional[str]:
        return cls.api_key

    @classmethod
    def set_timeout(cls, timeout_seconds: int) -> None:
        cls.timeout = timeout_seconds

    @classmethod
    def check_api_key(cls) -> None:
        if not cls.get_api_key():
            raise ValueError("No api_key initialized")

    @classmethod
    def chat_completition(cls, *args, **kwargs):
        # check for api key
        cls.check_api_key()

        open_ai_response: Optional[Dict[Any, Any]] = None

        # Add or overwrite timeout value in the kwargs
        kwargs['timeout'] = cls.timeout

        try:
            open_ai_response = openai.ChatCompletion.create(*args, **kwargs)
        except openai.error.Timeout:
            logger.warning("OpenAI call went in timeout")

        return open_ai_response
