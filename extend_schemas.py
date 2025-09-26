#!/usr/bin/env python3
"""
Extend SQL schemas in a folder by asking an Ollama model to add 10 tables,
then verify the combined schema is executable (SQLite) and save the result.

Usage:
  python extend_schemas.py --folder ./schemas --pattern "*.sql" \
      --model "Qwen3:30b" --retries 3

Defaults:
  folder  = current working directory
  pattern = "*.sql"
  model   = "Qwen3:30b"
  retries = 3
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import random
from pathlib import Path
from typing import Tuple

import ollama
from tqdm import tqdm

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def extract_sqlite_block(text: str) -> str | None:
    """
    Extracts the contents of the first ```sqlite ... ``` code block from a string.

    Args:
        text (str): The input string that may contain a sqlite fenced code block.

    Returns:
        str | None: The extracted SQL code inside the sqlite block,
                    or None if no block is found.
    """
    pattern = re.compile(r"```sqlite\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None


def check_executable_sqlite(schema_sql: str) -> Tuple[bool, str | None]:
    """
    Attempt to execute the provided SQL against an in-memory SQLite database.
    Returns (ok, error_message).
    """
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        conn.executescript(schema_sql)
        return True, None
    except sqlite3.Error as e:
        return False, str(e)
    finally:
        conn.close()


def build_initial_prompt(existing_schema: str) -> str:
    return (
        "/no_think"
        "Extend the following database schema with exactly 15 NEW tables.\n\n"
        "Requirements:\n"
        "1) Use **SQLite** dialect only. Avoid non-SQLite features (no ENUM, SERIAL, IDENTITY, MONEY, schemas, arrays, COMMENT ON, etc.).\n"
        "2) Keep existing objects unchanged. Only add new CREATE TABLE statements (plus any necessary CREATE INDEX statements).\n"
        "3) Each new table must have:\n"
        "   - A primary key (INTEGER PRIMARY KEY or TEXT primary key as appropriate).\n"
        "   - Sensible columns with types valid in SQLite (INTEGER, REAL, TEXT, BLOB, NUMERIC).\n"
        "   - Foreign keys where appropriate, referencing existing or newly added tables.\n"
        "4) Use similar naming schemes; keep names unique and consistent.\n"
        "5) Output executable Sqlite statements within ```sqlite ... ``` code blocks.\n"
        "6) Do not drop or alter existing tables.\n\n"
        "Existing schema:\n"
        f"{existing_schema}\n"
    )


def build_repair_prompt(existing_schema: str, last_error: str) -> str:
    return (
        "You previously produced SQL that failed to execute in SQLite. "
        "Produce a corrected version that **only** adds 15 new tables and is fully executable in SQLite.\n\n"
        "Keep the same intent and relationships, but fix any issues that would break on SQLite "
        "(e.g., unsupported types/constraints/ALTERs, bad references, reserved words, missing commas, etc.).\n\n"
        "Constraints:\n"
        "- SQLite dialect only; only DDL statements.\n"
        "- Keep existing tables unchanged; only CREATE TABLE for the 15 new tables (and optional CREATE INDEX statements).\n\n"
        f"SQLite error to address:\n{last_error}\n\n"
        "Existing schema:\n"
        f"{existing_schema}\n"
    )


def call_ollama(model: str, prompt: str) -> str:
    """
    Use the Ollama Python client to generate content. Returns the raw text content.
    """
    resp = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior database engineer. "
                    "You output only valid SQLite DDL without any explanations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    # return raw content (caller will extract fenced sqlite block or fallback)
    return resp["message"]["content"]


def get_sql_from_generated(content: str) -> str | None:
    """
    Try to extract a ```sqlite code block from the model output. If none found,
    try any fenced block. Finally fall back to returning the whole content.
    """
    if not content:
        return None
    sqlite_block = extract_sqlite_block(content)
    if sqlite_block:
        return sqlite_block

    # try any fenced block ```...```
    m = re.search(r"```(?:[\w-]+)?\s*(.*?)\s*```", content, re.DOTALL)
    if m:
        return m.group(1).strip()

    # fallback to entire content
    return content.strip()


def count_tables(schema_sql: str) -> int:
    """Rudimentary count of CREATE TABLE occurrences in the SQL text."""
    if not schema_sql:
        return 0
    return len(re.findall(r"CREATE\s+TABLE\b", schema_sql, flags=re.IGNORECASE))


def extend_schema_for_file(
    path: Path,
    model: str,
    retries: int,
    out_dir: Path | None,
    in_place: bool,
    min_tables: int,
    max_tables: int,
) -> None:
    original = read_text(path).strip()
    combined_schema = original
    current_tables = count_tables(combined_schema)

    # determine random target for this schema
    target = random.randint(min_tables, max_tables)
    print(f"Target for {path.name}: {target} tables (starting at {current_tables})")

    # if already at or above target, just save a copy with suffix and return
    if current_tables >= target:
        target_dir = out_dir if out_dir else path.parent
        out_name = f"{path.stem}_{current_tables}.sql"
        out_path = target_dir / out_name
        write_text(out_path, combined_schema)
        print(f"ℹ️  {path.name}: already has {current_tables} tables. Saved -> {out_path}")
        return

    batch = 0
    # repeatedly add batches of 10 tables until we meet or exceed target
    while current_tables < target:
        batch += 1
        base_prompt = build_initial_prompt(combined_schema)
        raw = call_ollama(model, base_prompt)
        sql_block = get_sql_from_generated(raw)

        if not sql_block:
            print(f"⚠️  {path.name}: model returned no SQL for batch {batch}. Aborting.")
            break

        candidate = f"{combined_schema.rstrip()}\n\n-- Added by model (batch {batch})\n{sql_block.strip()}\n"
        ok, err = check_executable_sqlite(candidate)
        attempts = 1

        while not ok and attempts < retries:
            repair_prompt = build_repair_prompt(combined_schema, err or "")
            raw = call_ollama(model, repair_prompt)
            sql_block = get_sql_from_generated(raw)
            candidate = f"{combined_schema.rstrip()}\n\n-- Added by model (batch {batch} attempt {attempts+1})\n{sql_block.strip()}\n"
            ok, err = check_executable_sqlite(candidate)
            attempts += 1

        target_dir = out_dir if out_dir else path.parent

        if ok:
            combined_schema = candidate
            current_tables = count_tables(combined_schema)
            out_name = f"{path.stem}_{current_tables}.sql"
            out_path = target_dir / out_name
            write_text(out_path, combined_schema)
            print(f"✅ {path.name}: added 10 tables (now {current_tables}) -> {out_path.name}")
        else:
            # Save best-effort output and include the error at the top as a SQL comment
            failed_out = target_dir / f"{path.stem}_{current_tables}.failed.sql"
            combined_with_error = f"-- Execution failed: {err}\n{candidate}"
            write_text(failed_out, combined_with_error)
            print(f"⚠️  {path.name}: could not execute combined schema. Saved -> {failed_out}")
            print(f"    SQLite error: {err}")
            break

    # If requested, overwrite the original file with the fully-extended schema
    if in_place and current_tables >= target:
        write_text(path, combined_schema)
        print(f"🔁 {path.name}: in-place updated with {current_tables} tables.")


def process_folder(
    folder: Path,
    pattern: str,
    model: str,
    retries: int,
    out_dir: Path | None,
    in_place: bool,
    min_tables: int,
    max_tables: int,
) -> None:
    files = sorted(folder.glob(pattern))
    if not files:
        print(f"No files found in {folder} matching pattern {pattern!r}.")
        return

    for f in tqdm(files, desc="Processing schemas"):
        if not f.is_file():
            continue
        extend_schema_for_file(
            f,
            model=model,
            retries=retries,
            out_dir=out_dir,
            in_place=in_place,
            min_tables=min_tables,
            max_tables=max_tables,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extend SQL schemas with 10 new tables using an Ollama model.")
    p.add_argument("--folder", type=Path, default=Path("."), help="Folder containing schema files.")
    p.add_argument("--pattern", type=str, default="*.sql", help="Glob pattern to select schema files.")
    p.add_argument("--model", type=str, default="Qwen3:30b", help="Ollama model name to use.")
    p.add_argument("--retries", type=int, default=3, help="Maximum attempts per file (including the first).")
    p.add_argument("--out-dir", type=Path, default=None, help="Directory to write outputs (ignored with --in-place).")
    p.add_argument("--in-place", action="store_true", help="Overwrite the input files with the extended schemas.")
    p.add_argument("--min-tables", type=int, default=90, help="Minimum target tables per schema (inclusive).")
    p.add_argument("--max-tables", type=int, default=110, help="Maximum target tables per schema (inclusive).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # sanitize bounds
    min_t = max(0, int(args.min_tables))
    max_t = max(min_t, int(args.max_tables))

    process_folder(
        folder=args.folder,
        pattern=args.pattern,
        model=args.model,
        retries=max(1, args.retries),
        out_dir=args.out_dir,
        in_place=args.in_place,
        min_tables=min_t,
        max_tables=max_t,
    )


if __name__ == "__main__":
    main()
