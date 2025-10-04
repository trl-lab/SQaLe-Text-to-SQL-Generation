#!/usr/bin/env python3
"""
Generate semi-synthetic questions from SQL schemas using vLLM library directly
Runs inference in fixed-size batches of 32 prompts and shows overall progress.
"""

import os
import glob
import time
import csv
import json
import random
import argparse
from typing import List, Tuple
from tqdm import tqdm
import random

from vllm import LLM, SamplingParams
from ReFoRCE.utils import extract_code_blocks  # keep your existing helper

AVG_QUESTIONS_PER_SCHEMA = 10
BATCH_SIZE = 32  # always 32 at a time

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
            if q:
                questions.add(q)
    return list(questions)

def probability(j: int) -> float:
    """
    Probability mass function for number of joins j in {0,...,10}.
    - P(0) = 0.20
    - P(1) = 0.20
    - For j=2..10: decreases linearly to 0 at j=10.
    """
    if j not in range(0, 11):
        return 0.0

    if j in (0, 1):
        return 0.20

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

def build_prompt(schema_sql: str, examples: list[str], count: int) -> str:
    j = sample_j()

    prompt = SYSTEM_PREFIX + "\n\n"
    prompt += f"/no_think Generate {count} diverse questions for the following schema:\n"
    prompt += schema_sql + "\n\n"

    if examples:
        prompt += "Use these examples as inspiration in terms of style and complexity:\n"
        prompt += "\n".join(examples[:10]) + "\n\n"

    # Explicit difficulty instruction
    prompt += f"The questions should be so complex that you need {j} number of joins to solve them.\n\n"

    prompt += (
        f"Please provide {count} questions.\n"
        "Use the following format for your response and put it into the plaintext code box:\n"
        "```plaintext\n<First Question>\n<2nd Question>\n...\n<Nth Question>\n```"
    )
    return prompt

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
    parser.add_argument("--example_questions_file", type=str, default="data/examples.csv", help="CSV with 'nl_prompt' column")
    parser.add_argument("--model", type=str, required=True, help="Path or name of the model to load with vLLM")
    parser.add_argument("--out", type=str, default="questions.jsonl", help="Output JSONL with (schema_file, schema_sql, question)")
    parser.add_argument("--per_schema", type=int, default=AVG_QUESTIONS_PER_SCHEMA, help="Questions per schema")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens to generate")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling (default -1 = disabled)")
    args = parser.parse_args()

    print("Loading example questions...")
    exemplars = load_example_questions(args.example_questions_file)
    print(f"Loaded {len(exemplars)} example questions.")

    sql_files = sorted(glob.glob(os.path.join(args.schema_folder, "*.sql")))
    print(f"Found {len(sql_files)} schema files.")

    # Init vLLM engine once
    sampling = SamplingParams(
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        top_k=(args.top_k if args.top_k is not None else -1),
    )
    llm = LLM(model=args.model, max_model_len=21000, tensor_parallel_size=2)

    work: List[Tuple[str, str, List[str], int]] = []
    for file in sql_files:
        with open(file, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        random.shuffle(exemplars)
        current_exemplars = exemplars[:10]
        work.append((os.path.basename(file), schema_sql, current_exemplars, args.per_schema))

    total_schemas = len(work)
    written = 0

    # Overall progress bar across ALL schemas (not just batches)
    with open(args.out, "w", encoding="utf-8") as fout:
        # Keep the batch bar for visibility, but correct the description
        for batch in tqdm(list(chunked(work, BATCH_SIZE)), desc=f"Generating (batches of {BATCH_SIZE})", leave=False):
            prompts = [build_prompt(schema_sql, exemplars, per_schema) for (_, schema_sql, exemplars, per_schema) in batch]
            try:
                outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                for (schema_file, schema_sql, _, per_schema), out in zip(batch, outputs):
                    raw = out.outputs[0].text if out.outputs else ""
                    candidates = parse_plaintext_block(raw)
                    questions = [q for q in candidates if q][:per_schema]
                    for q in questions:
                        rec = {
                            "schema_file": schema_file,
                            "schema_sql": schema_sql,
                            "question": q,
                            "source": "semi-synthetic",
                            "generation_model": args.model,
                            "timestamp": timestamp,
                        }
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1
            except Exception as e:
                continue

    print(f"Wrote {written} questions to {args.out}")

if __name__ == "__main__":
    main()