from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

# =============================
# LLM Adapter (pluggable)
# =============================
@dataclass
class GenerationConfig:
    temperature: float = 0.7         # sampling temperature
    top_p: float = 0.9               # nucleus sampling
    max_tokens: int = 512            # max tokens to generate

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
            temperature=cfg.temperature,
            top_p=cfg.top_p,
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
            candidates = [q.strip() for q in text.split("###") if q.strip()]
            results.append(candidates if candidates else [""])
        return results