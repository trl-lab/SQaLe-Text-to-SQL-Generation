from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple
from My_ReFoRCE.in_memory_db import InMemoryDB

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
    
class OpenAIAdapter(ModelAdapter):

    def __init__(self, model: str, api_key: str):
        super().__init__()
        self.model = model
        self.api_key = api_key

    def batch_generate(self, prompts: Sequence[str], system_prompt: Optional[str], cfg: GenerationConfig) -> List[List[str]]: