from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import pandas as pd
from vllm import LLM, SamplingParams
import asyncio  # if you need async support
from tqdm import tqdm

from routellm.routers.routers import ROUTER_CLS

# Default config for routers augmented using golden label data from GPT-4.
# This is exactly the same as config.example.yaml.
GPT_4_AUGMENTED_CONFIG = {
    "sw_ranking": {
        "arena_battle_datasets": [
            "lmsys/lmsys-arena-human-preference-55k",
            "routellm/gpt4_judge_battles",
        ],
        "arena_embedding_datasets": [
            "routellm/arena_battles_embeddings",
            "routellm/gpt4_judge_battles_embeddings",
        ],
    },
    "causal_llm": {"checkpoint_path": "routellm/causal_llm_gpt4_augmented"},
    "bert": {"checkpoint_path": "routellm/bert_gpt4_augmented"},
    "mf": {"checkpoint_path": "routellm/mf_gpt4_augmented"},
}


class RoutingError(Exception):
    pass


@dataclass
class ModelPair:
    strong: str
    weak: str


class Controller:
    def __init__(
        self,
        routers: list[str],
        strong_model: str,
        weak_model: str,
        config: Optional[dict[str, dict[str, Any]]] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        progress_bar: bool = False,
    ):
        self.model_pair = ModelPair(strong=strong_model, weak=weak_model)
        self.routers = {}
        self.api_base = api_base
        self.api_key = api_key
        self.model_counts = defaultdict(lambda: defaultdict(int))
        self.progress_bar = progress_bar

        if config is None:
            config = GPT_4_AUGMENTED_CONFIG

        router_pbar = None
        if progress_bar:
            router_pbar = tqdm(routers)
            tqdm.pandas()

        for router in routers:
            if router_pbar is not None:
                router_pbar.set_description(f"Loading {router}")
            self.routers[router] = ROUTER_CLS[router](**config.get(router, {}))

        # Mimic OpenAI's Chat Completions interface
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.completion, acreate=self.acompletion
            )
        )

    def _validate_router_threshold(
        self, router: Optional[str], threshold: Optional[float]
    ):
        if router is None or threshold is None:
            raise RoutingError("Router or threshold unspecified.")
        if router not in self.routers:
            raise RoutingError(
                f"Invalid router {router}. Available routers are {list(self.routers.keys())}."
            )
        if not 0 <= threshold <= 1:
            raise RoutingError(
                f"Invalid threshold {threshold}. Threshold must be a float between 0.0 and 1.0."
            )

    def _parse_model_name(self, model: str):
        _, router, threshold = model.split("-", 2)
        try:
            threshold = float(threshold)
        except ValueError as e:
            raise RoutingError(f"Threshold {threshold} must be a float.") from e
        if not model.startswith("router"):
            raise RoutingError(
                f"Invalid model {model}. Model name must be of the format 'router-[router name]-[threshold]."
            )
        return router, threshold

    def _get_routed_model_for_completion(
        self, messages: list, router: str, threshold: float
    ):
        # Use the last turn in the conversation for routing.
        # Our current routers were only trained on first turn data, so more research is required here.
        prompt = messages[-1]["content"]
        routed_model = self.routers[router].route(prompt, threshold, self.model_pair)
        self.model_counts[router][routed_model] += 1
        return routed_model

    # Mainly used for evaluations
    def batch_calculate_win_rate(
        self,
        prompts: pd.Series,
        router: str,
    ):
        self._validate_router_threshold(router, 0)
        router_instance = self.routers[router]
        if router_instance.NO_PARALLEL and self.progress_bar:
            return prompts.progress_apply(router_instance.calculate_strong_win_rate)
        elif router_instance.NO_PARALLEL:
            return prompts.apply(router_instance.calculate_strong_win_rate)
        else:
            return prompts.parallel_apply(router_instance.calculate_strong_win_rate)

    def route(self, prompt: str, router: str, threshold: float):
        self._validate_router_threshold(router, threshold)
        return self.routers[router].route(prompt, threshold, self.model_pair)

    def _build_prompt(self, messages: list) -> str:
        # Convert chat messages to a single prompt string.
        return "\n".join([msg["content"] for msg in messages])

    # Matches OpenAI's Chat Completions interface, but now using vLLM for completions.
    # Also supports optional router and threshold args.
    # If model name is present, attempt to parse router and threshold using it,
    # otherwise, use the router and threshold args.
    def completion(
        self,
        *,
        router: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])
        self._validate_router_threshold(router, threshold)
        routed_model = self._get_routed_model_for_completion(
            kwargs["messages"], router, threshold
        )
        
        print(f"Routed Model: {routed_model}")

        # prompt = self._build_prompt(kwargs["messages"])

        # # Create a vLLM instance for the routed model.
        # # Adjust max_model_len, trust_remote_code, and tensor_parallel_size as needed.
        # llm = LLM(routed_model, max_model_len=2048, trust_remote_code=True, tensor_parallel_size=1)
        # sampling_params = SamplingParams(
        #     temperature=kwargs.get("temperature", 1.0),
        #     top_p=kwargs.get("top_p", 0.9),
        #     max_tokens=kwargs.get("max_tokens", 100)
        # )
        # # Generate a completion using vLLM
        # result = llm.generate(prompt, sampling_params)
        return routed_model

    # Matches OpenAI's Async Chat Completions interface, but now using vLLM.
    # Since vLLM may not have native async support,
    # wrap the synchronous generate call in asyncio.to_thread.
    async def acompletion(
        self,
        *,
        router: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])
        self._validate_router_threshold(router, threshold)
        routed_model = self._get_routed_model_for_completion(
            kwargs["messages"], router, threshold
        )

        prompt = self._build_prompt(kwargs["messages"])
        llm = LLM(routed_model, max_model_len=2048, trust_remote_code=True, tensor_parallel_size=1)
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_tokens", 100)
        )
        # Wrap synchronous generation in asyncio.to_thread for async support.
        result = await asyncio.to_thread(llm.generate, prompt, sampling_params)
        return result, routed_model
