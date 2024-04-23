# encoding: utf-8
import os
import json
from time import sleep
import logging
from typing import List, Any

from azure.identity import AzureCliCredential, get_bearer_token_provider
import semantic_kernel as sk
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.exceptions import KernelInvokeException, FunctionExecutionException, ServiceResponseException
from openai import RateLimitError
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_fixed, RetryError

from correction.auth import get_cognitiveservices_access_token
from correction.definitions import Mistake, ReACTStep, MetaEvaluator
from correction.logging import log_attempt_number
from correction.wait import wait_log_extract_seconds


class SKEngine:
    def __init__(self, log: logging.Logger, skill_folder: str = "correction") -> None:
        self.log = log
        self.skill_folder = skill_folder
        self.get_azure_env()
        self.refresh_token_and_reload_service()

    def get_azure_env(self):
        self.user = os.environ.get("AZURE_OPENAI_USER")
        self.azure_settings = dict(
            deployment_name = os.environ.get("AZURE_DEPLOYMENT_NAME"),
            endpoint = os.environ.get("AZURE_ENDPOINT"),
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
        )

    def refresh_token_and_reload_service(self):
        self.kernel = sk.Kernel(log=self.log)
        token_provider = get_bearer_token_provider(AzureCliCredential(), "https://cognitiveservices.azure.com/.default")
        self.kernel.remove_all_services()
        settings = self.azure_settings
        oai_text_service = AzureChatCompletion(
            **settings,
            service_id="gpt",
            ad_token_provider=token_provider,
            default_headers={"x-ms-client-request-id": "DAXNLG", "user": self.user}
        )
        self.kernel.add_service(oai_text_service)
        self.skills = self.kernel.import_plugin_from_prompt_directory("./skills/", self.skill_folder)

    def get_context(self, **kwargs) -> KernelArguments:
        context = KernelArguments()
        for k, v in kwargs.items():
            if isinstance(v, str):
                context[k] = v
        return context

    async def run_skill(self, skill_name: str, context: KernelArguments) -> List[str]:
        skill = self.skills[skill_name]
        self.log.debug(skill.prompt_execution_settings)
        responses = []
        response = await self.kernel.invoke(plugin_name=self.skill_folder, function_name=skill_name, arguments=context)
        responses = [r.content for r in response.value]
        self.log.debug(responses)
        return responses

    @retry(
        stop=stop_after_attempt(30),
        wait=wait_log_extract_seconds(default_delay=15.0),
        retry=retry_if_exception_type((RateLimitError, KernelInvokeException, FunctionExecutionException, ServiceResponseException)),
        before_sleep=log_attempt_number
    )
    async def _run_engine(self, skill_name: str, **kwargs) -> List[str]:
        context = self.get_context(**kwargs)
        responses = await self.run_skill(skill_name, context)
        for i, r in enumerate(responses):
            self.log.debug(f"{i} - {r}")
        output = self.parse_output(responses)
        return output

    def parse_output(self, output: Any) -> List[str]:
        if not isinstance(output, list):
            output = [output]
        return output

    def _gen_postprocess(self, output: str) -> str:
        remove_items = ["```", "json", "OUTPUT:"]
        for item in remove_items:
            output = output.replace(item, "")
        return output.strip()

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(1),
        retry=retry_if_exception_type((ValueError, RetryError)),
        before_sleep=log_attempt_number
    )
    async def find_and_correct(self, skill_name: str = "find_and_correct", **kwargs) -> List[Mistake]:
        assert "clinical_note" in kwargs and isinstance(kwargs["clinical_note"], str), "Must provide clinical_note as a string"
        results = await self._run_engine(skill_name, **kwargs)
        self.log.debug(f"find_and_correct: {results}")
        results = map(self._gen_postprocess, results)
        outputs = [Mistake.from_str(o) for o in results]
        return outputs

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(30),
        retry=retry_if_exception_type((ValueError, RetryError)),
        before_sleep=log_attempt_number
    )
    async def correct_reasoning(self, skill_name: str = "correct_reasoning", **kwargs) -> List[Mistake]:
        assert "clinical_note" in kwargs and isinstance(kwargs["clinical_note"], str), "Must provide clinical_note as a string"
        results = await self._run_engine(skill_name, **kwargs)
        self.log.debug(f"correct_reasoning: {results}")
        results = map(self._gen_postprocess, results)
        return [str(r) for r in results]

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(30),
        retry=retry_if_exception_type((json.JSONDecodeError, RetryError)),
        before_sleep=log_attempt_number
    )
    async def react(self, skill_name: str = "react_step", **kwargs) -> list[ReACTStep]:
        assert "clinical_note" in kwargs and isinstance(kwargs["clinical_note"], str), "Must provide clinical_note as a string"
        assert "history" in kwargs and isinstance(kwargs["history"], str), "Must provide history as a string"
        results = await self._run_engine(skill_name, **kwargs)
        self.log.debug(f"react: {results}")
        results = map(self._gen_postprocess, results)
        outputs = [ReACTStep.from_str(o) for o in results]
        return outputs

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(30),
        retry=retry_if_exception_type((json.JSONDecodeError, RetryError)),
        before_sleep=log_attempt_number
    )
    async def evaluation(self, skill_name: str = "evaluate", **kwargs) -> MetaEvaluator:
        assert "clinical_note" in kwargs and isinstance(kwargs["clinical_note"], str), "Must provide clinical_note as a string"
        assert "search_history" in kwargs and isinstance(kwargs["search_history"], str), "Must provide history as a string"
        results = await self._run_engine(skill_name, **kwargs)
        self.log.debug(f"evaluation: {results}")
        results = map(self._gen_postprocess, results)
        outputs = MetaEvaluator.from_list_str(results)
        return outputs

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(30),
        retry=retry_if_exception_type((ValueError, RetryError)),
        before_sleep=log_attempt_number
    )
    async def reflexion(self, skill_name: str = "reflexion", **kwargs) -> List[str]:
        assert "clinical_note" in kwargs and isinstance(kwargs["clinical_note"], str), "Must provide clinical_note as a string"
        assert "search_history" in kwargs and isinstance(kwargs["search_history"], str), "Must provide history as a string"
        assert "reviews" in kwargs and isinstance(kwargs["reviews"], str), "Must provide reviews as a string"
        results = await self._run_engine(skill_name, **kwargs)
        self.log.debug(f"reflexion: {results}")
        results = map(self._gen_postprocess, results)
        return [str(r) for r in results]

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(30),
        retry=retry_if_exception_type((ValueError, RetryError)),
        before_sleep=log_attempt_number
    )
    async def error_classify(self, skill_name: str = "error_classify", **kwargs) -> List[str]:
        assert "error_sentence" in kwargs and isinstance(kwargs["error_sentence"], str), "Must provide error_sentence as a string"
        assert "correct_sentence" in kwargs and isinstance(kwargs["correct_sentence"], str), "Must provide correct_sentence as a string"
        results = await self._run_engine(skill_name, **kwargs)
        self.log.debug(f"error_classify: {results}")
        results = map(self._gen_postprocess, results)
        return [str(r) for r in results]

