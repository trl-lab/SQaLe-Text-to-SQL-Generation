#!/usr/bin/env python3
"""
Generate semi-synthetic questions from SQL schemas using vLLM library directly
Runs inference in fixed-size batches (capped at 32) and shows overall progress.
Ensures example questions shown to the model match the required number of joins.

Now with retry logic:
- If a response fails to parse a plaintext code block (0 extracted lines), the script
  will retry that specific item up to --retries times with a stricter format reminder.
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
BATCH_SIZE = 32  # hard cap per vLLM call
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


# Check for repeated template strings in the candidates
def has_template_strings(strings: List[str]) -> bool:
    templates = {
        "<First Question>",
        "<2nd Question>",
        "<Nth Question>",
        "```",
        "<3rd Question>",
        "<4th Question>",
        "<5th Question>",
        "plaintext",
        "<Second Question>",
        "<Fifth Question>",
        "<|Start of a new question|>",
    }
    for s in strings:
        for t in templates:
            if t in s:
                return True
    return False

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

    # Add task diversity instruction
    prompt += (
        "Vary the types of tasks you ask for (e.g., retrieval, aggregation, filtering, sorting, grouping, average etc.). One question could also require multiple such operations. "
        "Make sure the questions are realistic and relevant to the schema provided. "
        "Do not repeat or paraphrase the examples.\n\n"
    )

    # 50/50 chance of adding explicit instruction for schema and value-awareness. If not, the model is prompted to be vague
    if random.random() < 0.3:
        prompt += (
            "Make sure the questions are specific to the schema provided, "
            "and use table/column names. Also try to incorporate specific values that could exist in the schema.\n\n"
        )
    else:
        prompt += (
            "The questions should be more vague and do not need to reference specific table/column names or values.\n\n"
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

def build_retry_prompt(base_prompt: str, attempt_idx: int) -> str:
    """Append a progressively stricter format reminder for retries."""
    # attempt_idx: 1-based for readability in the appended note
    note = (
        "\n\n# FORMAT REMINDER (retry {n})\n"
        "Return ONLY one code fence labeled exactly ```plaintext ... ``` enclosing the questions, "
        "with one question per line. No extra text before or after the fence. "
        "Do not include notes, explanations, or any other markdown.".format(n=attempt_idx)
    )
    return base_prompt + note

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
    parser.add_argument("--retries", type=int, default=3, help="Max retry rounds for items that fail to parse a plaintext code block")
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
    hard_cap = min(int(args.max_batch_size), BATCH_SIZE)

    with open(args.out, "w", encoding="utf-8") as fout:
        # Process in batches for efficiency
        for batch in tqdm(list(chunked(work, hard_cap)), desc=f"Generating (batches of {hard_cap})", leave=False):
            # Build initial prompts & meta
            prompts_meta = [build_prompt(schema_sql, examples_by_j, per_schema) for (_, schema_sql, per_schema) in batch]
            base_prompts = [pm[0] for pm in prompts_meta]
            target_js = [pm[1] for pm in prompts_meta]
            examples_js = [pm[2] for pm in prompts_meta]

            # Track which indices are still pending due to parse failure
            pending = list(range(len(batch)))
            # Store the last raw text for debugging (optional)
            last_raw: Dict[int, str] = {}

            # Try initial + retries
            for attempt in range(0, args.retries + 1):
                if not pending:
                    break

                # Prepare prompts for this attempt: original on attempt 0, stricter thereafter
                attempt_prompts = []
                for idx in pending:
                    p = base_prompts[idx]
                    if attempt > 0:
                        p = build_retry_prompt(p, attempt)
                    attempt_prompts.append(p)

                try:
                    outputs = llm.generate(attempt_prompts, sampling_params=sampling, use_tqdm=False)
                except Exception as e:
                    print(f"[WARN] Batch attempt {attempt} failed with error: {e}")
                    # On a hard vLLM error, move to next attempt for all pending
                    continue

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                # Evaluate results; keep those that parsed, retain others in 'pending'
                next_pending = []
                for sub_i, idx in enumerate(pending):
                    (schema_file, schema_sql, per_schema) = batch[idx]
                    target_j = target_js[idx]
                    examples_j = examples_js[idx]

                    out = outputs[sub_i]
                    raw = out.outputs[0].text if out.outputs else ""
                    last_raw[idx] = raw

                    candidates = parse_plaintext_block(raw)

                    if has_template_strings(candidates):
                        # treat as parse failure, keep for retry
                        next_pending.append(idx)
                        continue

                    # Consider it a parse failure ONLY if we got zero lines.
                    if len(candidates) == 0:
                        # keep for another retry
                        next_pending.append(idx)
                        continue

                    # success path
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

                pending = next_pending

            # If anything is still pending after retries, log a warning but keep going
            if pending:
                for idx in pending:
                    schema_file = batch[idx][0]
                    print(f"[WARN] Could not parse plaintext block for {schema_file} after {args.retries} retries.")

    print(f"Wrote {written} questions to {args.out}")

if __name__ == "__main__":
    main()
