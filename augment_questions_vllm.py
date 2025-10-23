#!/usr/bin/env python3
"""Augment questions in a metadata-enriched JSONL dataset with paraphrased variants.

This script expects the input JSONL to match the structure produced by ``add_metadata.py``:
each line is a JSON object containing (at least) ``question``, ``query``, and ``schema`` fields,
plus optional metadata such as ``token_count``. For every record, the script:

1. Writes the original record to the output JSONL (unless ``--skip-original`` is supplied).
2. Builds an LLM prompt with the original question, its SQL query, and instructions to produce
   alternative phrasings that preserve intent.
3. Calls a vLLM model in configurable batches to generate paraphrases, enforcing a plaintext
   code-block response for easy parsing.
4. Appends a copy of the original record for each paraphrase with updated question text and
   token counts, along with augmentation metadata (``augmentation_type``, ``augmentation_of``
   etc.).

Example usage:

    python augment_questions_vllm.py \
        --input data_with_metadata.jsonl \
        --output data_with_paraphrases.jsonl \
        --model your-vllm-model \
        --num-alternatives 3

The output file will contain all original records (unless skipped) followed by the generated
paraphrased variants, sharing the same schema and SQL query fields.
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from vllm import LLM, SamplingParams

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None  # type: ignore

from ReFoRCE.utils import extract_code_blocks  # re-use shared helper
from add_metadata import count_tokens  # ensure consistent token accounting

BATCH_CAP = 32
DEFAULT_BATCH_SIZE = 12
DEFAULT_NUM_ALTERNATIVES = 3
SYSTEM_PREFIX = (
    "You are an expert at paraphrasing natural-language questions for SQL tasks. "
    "Produce faithful, diverse rewrites that keep the exact analytical intent of the provided SQL query."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paraphrased variants of questions in an add_metadata-enriched JSONL using vLLM. "
            "Original records are kept and paraphrases are appended with updated metadata."
        )
    )
    parser.add_argument("--input", required=True, help="Path to the JSONL produced by add_metadata.py")
    parser.add_argument("--output", required=True, help="Destination JSONL for originals + paraphrases")
    parser.add_argument("--model", required=True, help="Model name or path loadable by vLLM")
    parser.add_argument("--num-alternatives", type=int, default=DEFAULT_NUM_ALTERNATIVES,
                        help="Paraphrases to request per record (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Records per vLLM batch (capped at 32; default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum generation tokens")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling (default disables top-k)")
    parser.add_argument("--tensor-parallel-size", type=int, default=2,
                        help="Tensor parallel size for vLLM engine")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model context length")
    parser.add_argument("--retries", type=int, default=2,
                        help="Retries for responses that fail formatting checks (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic prompt sampling")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on the number of input records to process")
    parser.add_argument("--skip-original", action="store_true",
                        help="Do not copy original records to the output (only write paraphrases)")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress tracking even if tqdm is available")
    return parser.parse_args()


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def parse_plaintext_block(raw: str) -> List[str]:
    blocks = extract_code_blocks(raw, "plaintext")
    text = blocks[-1] if blocks else raw
    paraphrases: List[str] = []
    for line in text.splitlines():
        candidate = line.strip().strip("-•").strip()
        # Strip numbering or bullet prefixes that occasionally slip through.
        while candidate and (candidate[0].isdigit() or candidate[0] in {'.', ')'}):
            candidate = candidate[1:].lstrip()
        candidate = candidate.strip()
        if candidate:
            paraphrases.append(candidate)
    return paraphrases


def build_prompt(question: str, query: str, num_requested: int) -> str:
    prompt = SYSTEM_PREFIX + "\n\n"
    prompt += "Original question:\n"
    prompt += question.strip() + "\n\n"
    prompt += "SQL query that answers it:\n"
    prompt += "```sql\n" + query.strip() + "\n```\n\n"
    prompt += (
        f"Task: Provide {num_requested} alternative phrasings of the original question. "
        "Each paraphrase must stay fully faithful to the SQL query's intent, maintain the same requested outputs, "
        "and avoid adding new requirements. Use varied wording and sentence structure.\n\n"
        "Return format:\n"
        "```plaintext\nParaphrase 1\nParaphrase 2\n...\n```\n"
        "No commentary, no explanations, and no references to the paraphrasing process."
    )
    return prompt


def build_retry_prompt(base_prompt: str, attempt: int) -> str:
    return base_prompt + (
        "\n\n# FORMAT REMINDER (retry {n})\n"
        "Respond with exactly one fenced block labeled ```plaintext``` containing only the paraphrases, "
        "one per line. Remove bullets, numbering, or extra commentary."
    ).format(n=attempt)


def make_augmented_record(record: Dict[str, Any], paraphrase: str, model_name: str,
                          timestamp: str) -> Dict[str, Any]:
    augmented = copy.deepcopy(record)
    original_question = record.get("question", "")
    augmented["question"] = paraphrase

    token_count = augmented.get("token_count") or {}
    # Reuse existing query/schema token counts if present; otherwise recompute.
    query_tokens = token_count.get("query") if isinstance(token_count, dict) else None
    schema_tokens = token_count.get("schema") if isinstance(token_count, dict) else None

    if not isinstance(token_count, dict):
        token_count = {}

    token_count["question"] = count_tokens(paraphrase)
    token_count["query"] = query_tokens if query_tokens is not None else count_tokens(record.get("query", ""))
    token_count["schema"] = schema_tokens if schema_tokens is not None else count_tokens(record.get("schema", ""))
    token_count["total"] = token_count["question"] + token_count["query"] + token_count["schema"]
    augmented["token_count"] = token_count

    augmented["augmentation_type"] = "question_paraphrase"
    augmented["augmentation_of"] = original_question
    augmented["paraphrase_model"] = model_name
    augmented["paraphrased_at"] = timestamp
    return augmented


def process_batch(
    llm: LLM,
    sampling: SamplingParams,
    batch_entries: List[Tuple[Dict[str, Any], str]],
    num_alternatives: int,
    retries: int,
    model_name: str,
    out_stream,
    skip_duplicate: bool = True,
) -> int:
    if not batch_entries or num_alternatives <= 0:
        return 0

    base_prompts = [entry[1] for entry in batch_entries]
    pending = list(range(len(batch_entries)))
    paraphrase_map: Dict[int, List[str]] = {idx: [] for idx in range(len(batch_entries))}
    last_raw: Dict[int, str] = {}

    for attempt in range(retries + 1):
        if not pending:
            break
        prompts_for_attempt = []
        for idx in pending:
            prompt = base_prompts[idx]
            if attempt > 0:
                prompt = build_retry_prompt(prompt, attempt)
            prompts_for_attempt.append(prompt)

        try:
            outputs = llm.generate(prompts_for_attempt, sampling_params=sampling, use_tqdm=False)
        except Exception as exc:  # pragma: no cover - external engine failure
            print(f"[WARN] vLLM generation failed on attempt {attempt}: {exc}", file=sys.stderr)
            continue

        next_pending: List[int] = []
        for local_idx, global_idx in enumerate(pending):
            out = outputs[local_idx]
            raw = out.outputs[0].text if out.outputs else ""
            last_raw[global_idx] = raw
            paraphrases = parse_plaintext_block(raw)
            paraphrase_map[global_idx] = paraphrases
            if not paraphrases:
                next_pending.append(global_idx)
        pending = next_pending

    for idx in pending:
        entry = batch_entries[idx][0]
        original_question = entry.get("question", "")
        print(
            f"[WARN] Failed to parse paraphrases after retries for question: {original_question[:80]}",
            file=sys.stderr,
        )

    written = 0
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for idx, (record, _) in enumerate(batch_entries):
        paraphrases = paraphrase_map.get(idx, [])
        if not paraphrases:
            continue

        original_question = record.get("question", "")
        seen = {normalize(original_question)} if skip_duplicate else set()
        kept: List[str] = []
        for candidate in paraphrases:
            cleaned = candidate.strip()
            if not cleaned:
                continue
            norm = normalize(cleaned)
            if skip_duplicate and norm in seen:
                continue
            kept.append(cleaned)
            seen.add(norm)
            if len(kept) >= num_alternatives:
                break

        for paraphrase in kept:
            augmented = make_augmented_record(record, paraphrase, model_name, timestamp)
            out_stream.write(json.dumps(augmented, ensure_ascii=False) + "\n")
            written += 1

    return written


def stream_jsonl(path: str, limit: Optional[int] = None):
    with open(path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping malformed JSON line {idx + 1}: {exc}", file=sys.stderr)
                continue


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        print("[ERROR] Input and output paths must differ to avoid clobbering the source file.", file=sys.stderr)
        sys.exit(1)

    batch_size = min(max(1, args.batch_size), BATCH_CAP)
    sampling = SamplingParams(
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        top_k=(args.top_k if args.top_k is not None else -1),
    )

    llm = LLM(
        model=args.model,
        max_model_len=int(args.max_model_len),
        tensor_parallel_size=int(args.tensor_parallel_size),
    )

    progress_iter = stream_jsonl(args.input, limit=args.limit)
    if tqdm and not args.no_progress:
        # tqdm needs a known total; best-effort (may be None if limit enforced).
        total = args.limit
        progress_iter = tqdm(progress_iter, total=total, desc="Augmenting questions", unit="rec")  # type: ignore

    records_processed = 0
    paraphrases_written = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        batch: List[Tuple[Dict[str, Any], str]] = []
        for record in progress_iter:
            records_processed += 1

            if not args.skip_original:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            question = record.get("question", "")
            query = record.get("query", "")
            if not question or not query:
                print(
                    f"[WARN] Record {records_processed} missing question or query; skipping paraphrase.",
                    file=sys.stderr,
                )
                continue

            prompt = build_prompt(question, query, args.num_alternatives)
            batch.append((record, prompt))

            if len(batch) >= batch_size:
                paraphrases_written += process_batch(
                    llm=llm,
                    sampling=sampling,
                    batch_entries=batch,
                    num_alternatives=args.num_alternatives,
                    retries=args.retries,
                    model_name=args.model,
                    out_stream=fout,
                )
                batch = []

        if batch:
            paraphrases_written += process_batch(
                llm=llm,
                sampling=sampling,
                batch_entries=batch,
                num_alternatives=args.num_alternatives,
                retries=args.retries,
                model_name=args.model,
                out_stream=fout,
            )

    if tqdm and not args.no_progress:
        progress_iter.close()  # type: ignore[attr-defined]

    print(
        f"Processed {records_processed} records and wrote {paraphrases_written} paraphrased variants to {args.output}"
    )


if __name__ == "__main__":
    main()
