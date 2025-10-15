#!/usr/bin/env python3
"""
Generate semi-synthetic questions from SQL schemas using vLLM library directly
Runs inference in fixed-size batches of 32 prompts and shows overall progress.
Ensures example questions shown to the model match the required number of joins.
"""

import os
import glob
import time
import csv
import json
import random
import argparse
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import random

from vllm import LLM, SamplingParams
from ReFoRCE.utils import extract_code_blocks  # keep your existing helper

AVG_QUESTIONS_PER_SCHEMA = 10
BATCH_SIZE = 32  # always 32 at a time
NUMBER_OF_EXAMPLES = 4  # number of exemplar questions to include in the prompt (from the CSV file)

SYSTEM_PREFIX = (
    "You are an expert at writing diverse, realistic NL questions for SQL schemas. "
    "Answer ONLY with a plaintext code block. No explanations."
)

def load_example_questions(path: str) -> List[str]:
    questions = set()
    with open(path, "r", newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("nl_prompt", "").strip()
            number_of_joins = row.get("num_joins", "").strip()
            if q:
                questions.add((q, number_of_joins))
    return list(questions)

# ---------------------------
# New / updated helpers below
# ---------------------------

def normalize_examples(raw_examples: List[Tuple[str, str]]) -> List[Tuple[str, int]]:
    """Coerce num_joins to int; drop examples without valid integer join counts."""
    norm: List[Tuple[str, int]] = []
    for q, j in raw_examples:
        try:
            ji = int(j)
            if 0 <= ji <= 10:
                norm.append((q, ji))
        except Exception:
            continue
    return norm

def index_examples_by_joins(examples: List[Tuple[str, int]]) -> Dict[int, List[str]]:
    """Index example questions by their num_joins."""
    buckets: Dict[int, List[str]] = {j: [] for j in range(0, 11)}
    for q, j in examples:
        buckets.setdefault(j, []).append(q)
    return buckets

def probability(j: int) -> float:
    """
    Probability mass function for number of joins j in {0,...,10}.
    - P(0) = 0.15
    - P(1) = 0.25
    - For j=2..10: decreases linearly to 0 at j=10 with total mass summing to 1.
    """
    if j not in range(0, 11):
        return 0.0
    if j == 0:
        return 0.15
    elif j == 1:
        return 0.25
    # Linear decrease from j=2 to j=10
    m = 0.60 / 36  # slope chosen so total probability = 1
    return m * (10 - j)

def sample_j() -> int:
    """Sample a number of joins j according to the probability distribution."""
    probs = [probability(j) for j in range(11)]
    r = random.random()
    cumulative = 0.0
    for j, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return j
    return 10  # fallback (in case of rounding issues)

def nearest_join_with_examples(target_j: int, buckets: Dict[int, List[str]]) -> Optional[int]:
    """Find the closest j' that has at least one example; returns None if none exist."""
    if buckets.get(target_j):
        return target_j
    diffs = sorted(range(0, 11), key=lambda x: (abs(x - target_j), x))
    for j in diffs:
        if buckets.get(j):
            return j
    return None

def select_examples_for_j(
    buckets: Dict[int, List[str]],
    j: int,
    k: int
) -> Tuple[int, List[str]]:
    """
    Pick up to k examples whose num_joins matches j; fall back to nearest join count that has examples.
    Returns (used_j, examples).
    """
    used_j = nearest_join_with_examples(j, buckets)
    if used_j is None:
        return j, []  # No examples at all
    pool = buckets[used_j]
    if not pool:
        return used_j, []
    if len(pool) <= k:
        return used_j, list(pool)
    # sample without replacement
    return used_j, random.sample(pool, k)

def build_prompt(schema_sql: str, examples_by_j: Dict[int, List[str]], count: int) -> Tuple[str, int, int]:
    """
    Build a prompt for the model.
    Returns (prompt, target_j, examples_j) where:
      - target_j is the required join count for generation,
      - examples_j is the join count used for the examples actually shown.
    """
    target_j = sample_j()
    examples_j, picked_examples = select_examples_for_j(examples_by_j, target_j, NUMBER_OF_EXAMPLES)

    prompt = SYSTEM_PREFIX + "\n\n"
    prompt += f"/no_think Generate {count} diverse questions for the following schema:\n"
    prompt += schema_sql.strip() + "\n\n"

    if picked_examples:
        prompt += (
            f"Use these examples as inspiration in terms of writing style. "
            f"Each example below is solvable with exactly {examples_j} JOIN(s):\n"
        )
        for ex in picked_examples:
            prompt += f"- {ex}\n"
        prompt += "\n"

    # Explicit difficulty instruction
    prompt += (
        f"The questions you generate must (!!) be so complex that they need exactly {target_j} JOIN(s) to solve. "
        "Only diverge from this requirement if the schema does not contain enough tables to support that many joins.\n\n"
    )

    prompt += (
        f"Please provide {count} questions.\n"
        "Use the following format for your response and put it into the plaintext code box:\n"
        "```plaintext\n<First Question>\n<2nd Question>\n...\n<Nth Question>\n```"
    )
    return prompt, target_j, examples_j

def parse_plaintext_block(raw: str) -> List[str]:
    blocks = extract_code_blocks(raw, "plaintext")
    text = blocks[-1] if blocks else raw
    out = []
    for line in text.splitlines():
        s = line.strip().strip("-•").strip()
        s = s.lstrip("0123456789").lstrip(".").lstrip(")").lstrip("-").strip()
        if s:
            out.append(s)
    return out

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    parser = argparse.ArgumentParser(description="Generate and store semi-synthetic questions from schemas using vLLM batches.")
    parser.add_argument("--schema_folder", type=str, default="data/statements", help="Folder with *.sql schemas")
    parser.add_argument("--example_questions_file", type=str, default="data/examples.csv", help="CSV with 'nl_prompt' and 'num_joins' columns")
    parser.add_argument("--model", type=str, required=True, help="Path or name of the model to load with vLLM")
    parser.add_argument("--out", type=str, default="questions.jsonl", help="Output JSONL with (schema_file, schema_sql, question)")
    parser.add_argument("--per_schema", type=int, default=AVG_QUESTIONS_PER_SCHEMA, help="Questions per schema")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens to generate")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling (default -1 = disabled)")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="vLLM tensor parallel size")
    parser.add_argument("--max_batch_size", type=int, default=15, help="Max batch size for vLLM (will be capped at 32)")
    args = parser.parse_args()

    print("Loading example questions...")
    raw_exemplars = load_example_questions(args.example_questions_file)
    normalized_exemplars = normalize_examples(raw_exemplars)
    examples_by_j = index_examples_by_joins(normalized_exemplars)
    total_examples = sum(len(v) for v in examples_by_j.values())
    print(f"Loaded {total_examples} valid example questions with join labels.")

    sql_files = sorted(glob.glob(os.path.join(args.schema_folder, "*.sql")))
    # Remove files that contain "failed" in their name
    sql_files = [f for f in sql_files if "failed" not in os.path.basename(f).lower()]
    
    print(f"Found {len(sql_files)} schema files.")

    # Init vLLM engine once
    sampling = SamplingParams(
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        top_k=(args.top_k if args.top_k is not None else -1),
    )
    llm = LLM(model=args.model, max_model_len=21000, tensor_parallel_size=int(args.tensor_parallel_size))

    work: List[Tuple[str, str, int]] = []
    for file in sql_files:
        with open(file, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        work.append((os.path.basename(file), schema_sql, args.per_schema))

    written = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for batch in tqdm(list(chunked(work, args.max_batch_size)), desc=f"Generating (batches of {args.max_batch_size})", leave=False):
            prompts_meta = [build_prompt(schema_sql, examples_by_j, per_schema) for (_, schema_sql, per_schema) in batch]
            prompts = [pm[0] for pm in prompts_meta]
            target_js = [pm[1] for pm in prompts_meta]
            examples_js = [pm[2] for pm in prompts_meta]

            try:
                outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                for (schema_file, schema_sql, per_schema), out, target_j, examples_j in zip(batch, outputs, target_js, examples_js):
                    raw = out.outputs[0].text if out.outputs else ""
                    candidates = parse_plaintext_block(raw)
                    questions = [q for q in candidates if q][:per_schema]
                    for q in questions:
                        rec = {
                            "schema_file": schema_file,
                            "schema_sql": schema_sql,
                            "question": q,
                            "required_num_joins": target_j,     # what we asked the model to target
                            "examples_num_joins": examples_j,   # the join count of examples shown
                            "source": "semi-synthetic",
                            "generation_model": args.model,
                            "timestamp": timestamp,
                        }
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1
            except Exception as e:
                # Keep going on individual batch failures
                print(f"[WARN] Batch failed with error: {e}")
                continue

    print(f"Wrote {written} questions to {args.out}")

if __name__ == "__main__":
    main()
