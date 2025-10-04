#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple, Set, Iterable
import time

from tqdm import tqdm

from vllm import LLM
from My_ReFoRCE.model import VLLMAdapter, GenerationConfig
from My_ReFoRCE.sql import text2sql

def annotate_sql_type(sql_text: str) -> str:
    t = (sql_text or "").strip().upper()
    for kw in ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"):
        if t.startswith(kw):
            return kw
    return "OTHER"

def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def chunked(seq: List, size: int) -> Iterable[List]:
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def load_items(questions_file: str, dedup: bool = True) -> List[Tuple[str, str, str]]:
    """
    Load (prompt, schema, schema_file) items from questions.jsonl produced by generate_questions.py.
    Dedups by (schema_file, question).
    Returns a list of triples so we can carry schema_file into the final output if desired.
    """
    seen: Set[Tuple[str, str]] = set()
    items: List[Tuple[str, str, str]] = []

    for rec in iter_jsonl(questions_file):
        schema_file = rec.get("schema_file", "")
        schema_sql = (rec.get("schema_sql") or "").strip()
        question = (rec.get("question") or "").strip()
        if not schema_sql or not question:
            continue

        key = (schema_file, question)
        if dedup and key in seen:
            continue
        seen.add(key)

        items.append((question, schema_sql, schema_file))
    return items

def main():
    parser = argparse.ArgumentParser(description="Batch Text2SQL: load stored questions and produce SQL with vLLM.")
    parser.add_argument("--questions_file", type=str, default="questions.jsonl",
                        help="Input JSONL from generate_questions.py")
    parser.add_argument("--out", type=str, default="semi_synthetic_dataset.jsonl",
                        help="Output JSONL with prompt/sql/schema/cmd_type")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B-FP8",
                        help="vLLM model name or local path (default: Qwen/Qwen3-14B-FP8)")
    parser.add_argument("--tp", type=int, default=int(os.environ.get("TP_SIZE", "1")),
                        help="Tensor parallel size (default: 1)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="How many (prompt,schema) pairs to process per text2sql call (default: 16)")
    parser.add_argument("--candidates", type=int, default=int(os.environ.get("CANDIDATES_PER_ITEM", "2")),
                        help="Number of single-candidate generations per item for voting (default: 2)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for candidate generation (default: 1.0)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p nucleus sampling for candidate generation (default: 0.95)")
    parser.add_argument("--vote_temperature", type=float, default=0.0,
                        help="Temperature for vote/critique passes (default: 0.0)")
    parser.add_argument("--vote_top_p", type=float, default=1.0,
                        help="Top-p for vote/critique passes (default: 1.0)")
    parser.add_argument("--max_model_len", type=int, default=21000,
                        help="Max model context length for vLLM (default: 21000)")
    args = parser.parse_args()

    # 1) Load items
    items_all: List[Tuple[str, str, str]] = load_items(args.questions_file, dedup=True)
    if not items_all:
        print("No valid (question, schema) pairs found. Exiting.")
        return

    # 2) Spin up vLLM + adapter once
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
    )
    adapter = VLLMAdapter(model=llm)

    cfg_generate = GenerationConfig(temperature=args.temperature, top_p=args.top_p)
    cfg_vote = GenerationConfig(temperature=args.vote_temperature, top_p=args.vote_top_p)

    # 3) Batch over items
    total = len(items_all)
    written = 0

    current_timestamp = time.time()

    with open(args.out, "w", encoding="utf-8") as fout:
        pbar = tqdm(total=total, desc="Generating SQL (batched)")

        for batch in chunked(items_all, args.batch_size):
            # Prepare (prompt, schema) pairs for this batch
            items_ps: List[Tuple[str, str]] = [(q, s) for (q, s, _schema_file) in batch]

            # Call ReFoRCE text2sql once for the batch
            sqls: List[str] = text2sql(
                items=items_ps,
                adapter=adapter,
                cfg_generate=cfg_generate,
                cfg_vote=cfg_vote,
                candidates_per_item=args.candidates,
            )

            # Write aligned outputs
            for (question, schema_sql, _schema_file), sql in zip(batch, sqls):
                sql = (sql or "").strip()
                record = {
                    "prompt": " ".join(question.splitlines()).strip(),
                    "sql_statement": " ".join(sql.splitlines()).strip(),
                    "schema": " ".join(schema_sql.splitlines()).strip(),
                    "cmd_type": annotate_sql_type(sql),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            pbar.update(args.batch_size)

        pbar.close()

    time_end = time.time()
    print(f"Time taken: {time_end - current_timestamp:.2f} seconds")
    print(f"Processed {total} items; wrote {written} prompt-SQL pairs to {args.out}")

if __name__ == "__main__":
    main()