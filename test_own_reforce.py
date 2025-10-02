# test_text2sql_qwen.py
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

from My_ReFoRCE.in_memory_db import InMemoryDB
from My_ReFoRCE.model import VLLMAdapter, GenerationConfig, AsyncOpenAIAdapter
from My_ReFoRCE.sql import text2sql
from dotenv import load_dotenv

def build_items() -> List[Tuple[str, str]]:
    """Create a small suite of (prompt, schema) pairs to exercise the batch path."""
    ddl_1 = """
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        name TEXT,
        country TEXT,
        created_at TEXT
    );
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        amount REAL,
        status TEXT,
        created_at TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """
    prompt_1 = "List the top 3 countries by number of users, descending."

    ddl_2 = """
    CREATE TABLE movies (
        id INTEGER PRIMARY KEY,
        title TEXT,
        year INTEGER,
        rating REAL
    );
    CREATE TABLE reviews (
        id INTEGER PRIMARY KEY,
        movie_id INTEGER,
        stars INTEGER,
        FOREIGN KEY(movie_id) REFERENCES movies(id)
    );
    """
    prompt_2 = "Show the average movie rating per year after 2015, sorted descending, limit 5."

    ddl_3 = """
    CREATE TABLE departments (
        id INTEGER PRIMARY KEY,
        name TEXT
    );
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        department_id INTEGER,
        salary REAL,
        FOREIGN KEY(department_id) REFERENCES departments(id)
    );
    """
    prompt_3 = "Compute total salary per department, highest first."

    return [
        (prompt_1, ddl_1),
        (prompt_2, ddl_2),
        (prompt_3, ddl_3),
    ]

def main(model_name: str, tensor_parallel_size: int, candidates_per_item: int):
    from vllm import LLM

    """llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=15000
    )
    adapter = VLLMAdapter(model=llm)"""

    load_dotenv()

    adapter = AsyncOpenAIAdapter(
        model="gpt-5-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        max_concurrency=8,
        max_retries=5,
        base_delay=0.5,
    )

    items = build_items()

    sqls = text2sql(
        items=items,
        adapter=adapter,
        cfg_generate=GenerationConfig(temperature=1.0, top_p=0.95),
        cfg_vote=GenerationConfig(temperature=0.0, top_p=1.0),
        candidates_per_item=candidates_per_item,
    )

    print("\n=== Text2SQL Results ===")
    for i, ((prompt, _schema), sql) in enumerate(zip(items, sqls), start=1):
        print(f"\nCase {i}:")
        print(f"Prompt: {prompt}")
        print("SQL:")
        print(sql)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-14B-FP8",
        help="vLLM model name or local path (default: Qwen/Qwen3-14B-FP8)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=int(os.environ.get("TP_SIZE", "1")),
        help="Tensor parallel size (default: 1)",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=int(os.environ.get("CANDIDATES_PER_ITEM", "2")),
        help="Number of single-candidate generations per item (default: 2)",
    )
    args = parser.parse_args()

    main(
        model_name=args.model,
        tensor_parallel_size=args.tp,
        candidates_per_item=args.candidates,
    )