from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple
import asyncio
import math

# openbatch imports
from openai import AsyncOpenAI
from openai import RateLimitError, APIError, APITimeoutError, InternalServerError

# =============================
# LLM Adapter (pluggable)
# =============================
@dataclass
class GenerationConfig:
    temperature: float = 0.7         # sampling temperature
    top_p: float = 0.9               # nucleus sampling
    max_tokens: int = 4096            # max tokens to generate

class ModelAdapter:
    """Abstract adapter. Implement batch_generate to return k candidates per prompt.
    Returns: list[list[str]] with shape [len(prompts)][num_votes].
    """
    def batch_generate(self, prompts: Sequence[str], system_prompt: Optional[str], cfg: GenerationConfig) -> List[List[str]]:
        raise NotImplementedError

class VLLMAdapter(ModelAdapter):
    """Example adapter for vLLM (https://github.com/vllm-project/vllm)"""
    def __init__(self, model: Any):
        self.model = model

    def batch_generate(self, prompts: Sequence[str], system_prompt: Optional[str], cfg: GenerationConfig) -> List[List[str]]:
        from vllm import LLM, SamplingParams

        sampling_params = SamplingParams(
            max_tokens=cfg.max_tokens,
            top_k=-1,  # disable top-k
        )

        full_prompts = []
        for prompt in prompts:
            if system_prompt:
                full_prompts.append(f"{system_prompt}\n\n{prompt}")
            else:
                full_prompts.append(prompt)

        outputs = self.model.generate(full_prompts, sampling_params=sampling_params, use_tqdm=False)
        results = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            results.append(text)
        return results



class AsyncOpenAIAdapter(ModelAdapter):
    """
    Fully-async adapter using the classic OpenAI Chat Completions API.

    - Processes all prompts concurrently with asyncio.gather
    - Concurrency limited via `max_concurrency`
    - Simple retry/backoff on transient errors
    - Returns one candidate per prompt (same shape as other adapters)

    If you *must* keep the sync interface, see `run_sync` at the bottom.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        max_concurrency: int = 8,
        max_retries: int = 5,
        base_delay: float = 0.5,  # seconds for backoff start
    ):
        super().__init__()
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)
        # Don't create an asyncio.Semaphore here because it would be bound
        # to the event loop that constructs the adapter. Instead, store the
        # concurrency value and create a semaphore inside the active
        # async context (see batch_generate_async).
        self._max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.base_delay = base_delay

    # -------------- public async API --------------
    async def batch_generate_async(
        self,
        prompts: Sequence[str],
        system_prompt: Optional[str],
        cfg: GenerationConfig,
    ) -> List[List[str]]:
        # Create a semaphore bound to the currently running event loop.
        semaphore = asyncio.Semaphore(self._max_concurrency)

        tasks = [
            asyncio.create_task(self._generate_one(i, prompt, system_prompt, cfg, semaphore))
            for i, prompt in enumerate(prompts)
        ]
        results: List[Tuple[int, str]] = await asyncio.gather(*tasks)
        # restore original order by index
        results.sort(key=lambda x: x[0])
        return [[text] for _, text in results]

    # Keep the original sync signature if your code expects it.
    # It simply runs the async method to completion.
    def batch_generate(
        self,
        prompts: Sequence[str],
        system_prompt: Optional[str],
        cfg: GenerationConfig,
    ) -> List[List[str]]:
        return asyncio.run(self.batch_generate_async(prompts, system_prompt, cfg))

    # -------------- internals --------------
    async def _generate_one(
        self,
        idx: int,
        prompt: str,
        system_prompt: Optional[str],
        cfg: GenerationConfig,
        semaphore: asyncio.Semaphore,
    ) -> Tuple[int, str]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # retry with exponential backoff on transient errors
        attempt = 0
        while True:
            try:
                # Use the semaphore bound to the current event loop.
                async with semaphore:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                    )
                text = resp.choices[0].message.content or ""
                return idx, text

            except (RateLimitError, APITimeoutError, InternalServerError, APIError) as e:
                print(f"Error on prompt {idx}, attempt {attempt + 1}: {e}")
                attempt += 1
                if attempt > self.max_retries:
                    # give up and return best-effort info
                    return idx, f""
                # exponential backoff with jitter
                delay = self.base_delay * (2 ** (attempt - 1))
                delay = delay * (0.75 + 0.5 * (asyncio.get_running_loop().time() % 1))  # tiny jitter
                await asyncio.sleep(delay)

    # Optional: if you want an explicit sync runner separate from batch_generate
    def run_sync(
        self,
        prompts: Sequence[str],
        system_prompt: Optional[str],
        cfg: GenerationConfig,
    ) -> List[List[str]]:
        return asyncio.run(self.batch_generate_async(prompts, system_prompt, cfg))