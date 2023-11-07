import logging
from typing import Optional

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)


class OpenAiLibWrapper:

    api_key: Optional[str] = None
    client: AsyncOpenAI = None
    timeout: 0

    @classmethod
    def set_api_key(cls, api_key: str) -> None:
        if api_key is None:
            raise ValueError("api_key cannot be None.")
        cls.api_key = api_key
        # initialize open ai library
        cls.client = AsyncOpenAI(api_key=api_key)

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
    async def chat_completition(cls, *args, **kwargs) -> ChatCompletion:
        # check for api key
        cls.check_api_key()

        open_ai_response: ChatCompletion

        # Add or overwrite timeout value in the kwargs
        kwargs['timeout'] = cls.timeout

        try:
            open_ai_response = await cls.client.chat.completions.create(*args, **kwargs)
            return open_ai_response
        except openai.APITimeoutError:
            logger.error("OpenAI call went in timeout")
        except openai.APIConnectionError as e:
            logger.error("The server could not be reached")
            logger.error(e.__cause__)  # an underlying Exception, likely raised within httpx.
        except openai.RateLimitError as e:
            logger.error("A 429 status code was received; we should back off a bit.")
        except openai.APIStatusError as e:
            logger.error("Another non-200-range status code was received")
            logger.error(e.status_code)
            logger.error(e.response)
