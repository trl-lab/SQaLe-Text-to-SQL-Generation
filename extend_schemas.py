#!/usr/bin/env python3
"""
Extend SQL schemas in a folder by asking an LLM to add 15 tables,
then verify the combined schema is executable (SQLite) and save the result.

Rewritten to use the provided adapter classes (imported, unmodified):
- GenerationConfig
- ModelAdapter
- VLLMAdapter
- AsyncOpenAIAdapter

Usage examples:
  # vLLM backend (local engine managed outside this script)
  python extend_schemas_adapters.py --folder ./schemas --pattern "*.sql" \
      --backend vllm --model "Qwen/Qwen3-14B-FP8" --retries 3

  # OpenAI backend (async under the hood; adapter handles concurrency/retries)
  python extend_schemas_adapters.py --folder ./schemas --pattern "*.sql" \
      --backend openai --model "gpt-4o-mini" --retries 3 --openai-api-key sk-...

Defaults:
  folder      = current working directory
  pattern     = "*.sql"
  backend     = "vllm"
  model       = "Qwen/Qwen3-14B-FP8"
  retries     = 3
  min_tables  = 90
  max_tables  = 110
  top_k       = None (kept for CLI compatibility; not used by adapters here)
  temperature = 0.2
  max_tokens  = 4096
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import random
from dataclasses import dataclass
from pathlib import Path
import time
from typing import List, Tuple, Optional, Dict, Sequence
import numpy as np

from tqdm import tqdm

# ---------- import the provided classes (do not modify them) ----------
from My_ReFoRCE.model import (
    GenerationConfig,
    ModelAdapter,
    VLLMAdapter,
    AsyncOpenAIAdapter,
)

# ----------------------------- File I/O ---------------------------------

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ----------------------------- SQL helpers ------------------------------

def extract_sqlite_block(text: str) -> Optional[str]:
    """
    Extract the contents of the first ```sqlite ... ``` code block.
    If not found, try ```sql ... ```, then any code block.
    If still not found, extract from the first CREATE TABLE to the last ';'.
    """
    if not text:
        return None

    block = _extract_code_block(text, "sqlite")
    if block:
        return block

    block = _extract_code_block(text, "sql")
    if block:
        return block

    block = _extract_any_code_block(text)
    if block:
        return block

    block = _extract_from_create_table(text)
    if block:
        return block

    return text.strip()


def _extract_code_block(text: str, lang: str) -> Optional[str]:
    pattern = re.compile(rf"```{lang}\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    m = pattern.search(text)
    if m:
        return m.group(1).strip()
    return None


def _extract_any_code_block(text: str) -> Optional[str]:
    pattern = re.compile(r"```[\w-]*\s*(.*?)\s*```", re.DOTALL)
    m = pattern.search(text)
    if m:
        return m.group(1).strip()
    return None


def _extract_from_create_table(text: str) -> Optional[str]:
    create_match = re.search(r"(CREATE\s+TABLE\b.*?;)", text, re.IGNORECASE | re.DOTALL)
    if create_match:
        start = create_match.start()
        last_semicolon = text.rfind(";")
        if last_semicolon > start:
            return text[start:last_semicolon + 1].strip()
        else:
            return text[start:].strip()
    return None


def get_sql_from_generated(content: str) -> Optional[str]:
    """
    Prefer ```sqlite blocks, then any fenced block, else raw text.
    """
    if not content:
        return None
    b = extract_sqlite_block(content)
    if b:
        return b
    m = re.search(r"```(?:[\w-]+)?\s*(.*?)\s*```", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    return content.strip()


def check_executable_sqlite(schema_sql: str) -> Tuple[bool, Optional[str]]:
    """
    Try executing against in-memory SQLite.
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


def count_tables(schema_sql: str) -> int:
    if not schema_sql:
        return 0
    return len(re.findall(r"\bCREATE\s+TABLE\b", schema_sql, flags=re.IGNORECASE))


# ----------------------------- Prompting --------------------------------

SYSTEM_INSTRUCTION = (
    "You are a senior database engineer. "
    "You output only valid SQLite DDL without any explanations."
)

def build_initial_prompt(existing_schema: str, current_tables: int, target_tables: int) -> str:
    minimum = random.randint(15, 25)
    add_tables = min(minimum, target_tables - current_tables)
    return (
        "/no_think\n"
        f"Extend the following database schema with exactly {add_tables} NEW tables.\n\n"
        "Requirements:\n"
        "1) Use SQLite dialect only. Avoid non-SQLite features (no ENUM, SERIAL, IDENTITY, MONEY, schemas, arrays, COMMENT ON, etc.).\n"
        "2) Keep existing objects unchanged. Only add new CREATE TABLE statements (plus any necessary CREATE INDEX statements).\n"
        "3) Each new table should be a sensible and realistic extension to the existing schema.\n"
        "4) Use similar naming schemes, patterns and style.\n"
        "5) Output executable SQLite statements within ```sqlite ... ``` code blocks.\n"
        "6) Do not drop or alter existing tables. Make sure that there are no logical errors in the foreign key references.\n\n"
        "Existing schema:\n"
        f"{existing_schema}\n"
    )


def build_repair_prompt(existing_schema: str, last_error: str, current_tables: int, target_tables: int) -> str:
    minimum = random.randint(15, 25)
    add_tables = min(minimum, target_tables - current_tables)
    return (
        "You previously produced SQL that failed to execute in SQLite. "
        "Keep the same relationships, but avoid any issues that would break on SQLite "
        "(e.g., unsupported types/constraints/ALTERs, bad references, reserved words, missing commas, etc.).\n\n"
        "Constraints:\n"
        "- SQLite dialect only; only DDL statements.\n"
        f"- Keep existing tables unchanged; only CREATE TABLE for the {add_tables} new tables (and optional CREATE INDEX statements).\n\n"
        "- Output executable SQLite statements within ```sqlite ... ``` code blocks.\n\n"
        "- Do not drop or alter existing tables. Ensure no logical errors in foreign key references.\n"
        "Existing schema:\n"
        f"{existing_schema}\n"
    )

def get_probability_distribution(mode, max_val):
    """Return a smooth gamma-like density with given mode and cutoff at max_val.
    Before the peak, the curve starts at 0.5 and rises to 1 at the peak."""
    x = np.arange(0, max_val + 1)
    theta = max_val / 10
    k = max(mode / theta + 1, 1.5)  # ensure shape>1 for a defined mode
    y = (x ** (k - 1)) * np.exp(-x / theta)
    y = y / y.max()  # normalize so peak is 1

    # Find the peak index
    peak_idx = np.argmax(y)
    # Linearly interpolate from 0.5 at x_min to y[peak_idx] (which is 1) at the peak
    if peak_idx > 0:
        y[:peak_idx] = np.linspace(0.5, 1.0, peak_idx)
    # No noise added
    y[x > max_val] = 0  # strictly 0 above max_val
    y = np.clip(y, 0, None)
    y = y / y.sum()  # Normalize to sum to 1 for probability distribution
    return y


# ----------------------------- Job tracking -----------------------------

@dataclass
class Job:
    path: Path
    out_dir: Optional[Path]
    in_place: bool
    retries: int

    # dynamic
    combined_schema: str = ""
    current_tables: int = 0
    target: int = 0
    batch_num: int = 0
    attempts_for_current_round: int = 0
    last_error: Optional[str] = None
    finished: bool = False
    failed: bool = False
    # what to ask next (None if no prompt this round)
    next_prompt: Optional[str] = None
    is_repair: bool = False

    def init_from_file(self) -> None:
        original = read_text(self.path).strip()
        self.combined_schema = original
        self.current_tables = count_tables(self.combined_schema)
        distribution = get_probability_distribution(100, 350)
        self.target = random.choices(range(0, 351), weights=distribution)[0]
        print("" f"File {self.path.name}: {self.current_tables} tables, target {self.target} tables.")

    def prepare_initial_or_finish(self) -> None:
        if self.current_tables >= self.target:
            # Already at/over target: save copy and mark finished
            target_dir = self.out_dir if self.out_dir else self.path.parent
            out_name = f"{self.path.stem}_{self.current_tables}.sql"
            write_text(target_dir / out_name, self.combined_schema)
            self.finished = True
            return
        self.batch_num += 1
        self.attempts_for_current_round = 1
        self.is_repair = False
        self.next_prompt = build_initial_prompt(self.combined_schema, self.current_tables, self.target)

    def schedule_repair_or_fail(self) -> None:
        if self.attempts_for_current_round < self.retries:
            self.attempts_for_current_round += 1
            self.is_repair = True
            self.next_prompt = build_repair_prompt(self.combined_schema, self.last_error or "", self.current_tables, self.target)
        else:
            # Give up on this round/file
            target_dir = self.out_dir if self.out_dir else self.path.parent
            failed_out = target_dir / f"{self.path.stem}_{self.current_tables}.failed.sql"
            candidate_with_error = f"-- Execution failed: {self.last_error}\n{self.combined_schema}\n"
            write_text(failed_out, candidate_with_error)
            self.failed = True
            self.finished = True
            self.next_prompt = None

    def on_successful_addition(self, new_sql: str) -> None:
        # append the new block, write an intermediate
        self.combined_schema = f"{self.combined_schema.rstrip()}\n\n{new_sql.strip()}\n"
        self.current_tables = count_tables(self.combined_schema)

        # write intermediate file
        target_dir = self.out_dir if self.out_dir else self.path.parent
        out_name = f"{self.path.stem}_{self.current_tables}.sql"
        write_text(target_dir / out_name, self.combined_schema)

        # if target met, optionally overwrite original and finish; else enqueue next round's initial prompt
        if self.current_tables >= self.target:
            if self.in_place:
                write_text(self.path, self.combined_schema)
            self.finished = True
            self.next_prompt = None
        else:
            # next round initial prompt
            self.is_repair = False
            self.batch_num += 1
            self.attempts_for_current_round = 1
            self.next_prompt = build_initial_prompt(self.combined_schema, self.current_tables, self.target)


# ----------------------------- Adapter runner ---------------------------

def _normalize_adapter_output(raw: List) -> List[str]:
    """
    The provided adapters return different shapes:
      - AsyncOpenAIAdapter -> List[List[str]] (one candidate per prompt)
      - VLLMAdapter        -> List[str]      (per its current implementation)
    This function normalizes into List[str].
    """
    if not raw:
        return []
    # If it's already a list of strings, return as-is
    if isinstance(raw[0], str):
        return raw  # type: ignore
    # Else assume list[list[str]] and take first candidate
    out: List[str] = []
    for candidates in raw:  # type: ignore
        if isinstance(candidates, list) and candidates:
            out.append(candidates[0])
        else:
            out.append("")
    return out


def run_batched_round(adapter: ModelAdapter, prompts: List[str], cfg: GenerationConfig) -> List[str]:
    """
    Runs one batched round via the provided adapter and returns a list of raw model
    outputs (text) aligned to prompts.
    """
    raw = adapter.batch_generate(prompts, SYSTEM_INSTRUCTION, cfg)
    return _normalize_adapter_output(raw)  # ensure List[str]


# ----------------------------- Orchestration ----------------------------

def process_folder_batched(
    folder: Path,
    pattern: str,
    adapter: ModelAdapter,
    retries: int,
    out_dir: Optional[Path],
    in_place: bool,
    temperature: float,
    max_tokens: int,
    top_k: Optional[int],
    max_batch_size: int,
) -> None:
    """
    Batch-extend many SQL schemas in rounds using the provided adapter.

    Each round:
      - Gather up to `max_batch_size` active jobs that have a queued prompt.
      - Run a single batched generation call.
      - For each job, validate/commit or enqueue a repair prompt for the next round.
      - Jobs finish when they meet/exceed their per-file target or fail after retries.

    Writes intermediate outputs like: <name>_<tablecount>.sql
    On failure: <name>_<tablecount>.failed.sql with the last SQLite error at top.
    """
    files = sorted(folder.glob(pattern))
    if not files:
        print(f"No files found in {folder} matching pattern {pattern!r}.")
        return

    # Initialize jobs
    jobs: List[Job] = []
    for f in files:
        if not f.is_file():
            continue
        j = Job(
            path=f,
            out_dir=out_dir,
            in_place=in_place,
            retries=max(1, retries),
        )
        j.init_from_file()
        j.prepare_initial_or_finish()
        jobs.append(j)

    # Remove jobs that finished immediately (already at/over target)
    active: List[Job] = [j for j in jobs if not j.finished]
    if not active:
        print("Nothing to do. All schemas already at/above their targets.")
        return

    # Generation config (mapped to adapter semantics)
    gen_cfg = GenerationConfig(
        temperature=float(temperature),
        # top_p is part of GenerationConfig; not exposed via CLI in original script.
        # If desired, you can adjust here; we keep the class default.
        max_tokens=int(max_tokens),
    )

    round_idx = 0
    rr_cursor = 0  # round-robin cursor among *eligible* jobs for a round
    completed_names = set()

    start_time = time.time()

    with tqdm(total=len(active), desc="Schemas completed", unit="schema") as pbar:
        while True:
            # Eligible = have a prompt ready and not finished
            eligible: List[Job] = [j for j in active if (j.next_prompt is not None and not j.finished)]

            if not eligible:
                # Advance progress for any newly finished jobs
                newly_done = [j for j in active if j.finished and j.path.name not in completed_names]
                for _ in newly_done:
                    pbar.update(1)
                for j in newly_done:
                    completed_names.add(j.path.name)

                # If everyone is finished, exit
                if all(j.finished for j in active):
                    break

                # Otherwise, try to schedule a repair for stuck jobs
                for j in active:
                    if not j.finished and j.next_prompt is None:
                        j.schedule_repair_or_fail()
                # Next loop will re-evaluate "eligible"
                continue

            # Select a capped batch via round-robin for fairness
            n = len(eligible)
            if n <= max_batch_size:
                round_jobs = eligible
            else:
                start = rr_cursor % n
                end = start + max_batch_size
                if end <= n:
                    round_jobs = eligible[start:end]
                else:
                    round_jobs = eligible[start:] + eligible[: end - n]
                rr_cursor = (start + len(round_jobs)) % n

            round_idx += 1
            prompts = [j.next_prompt or "" for j in round_jobs]

            # Run single batched generation call via adapter
            raw_texts = run_batched_round(adapter, prompts, gen_cfg)

            # Clear queued prompts (they'll be replaced based on outcomes)
            for j in round_jobs:
                j.next_prompt = None

            # Handle each result
            for j, raw in zip(round_jobs, raw_texts):
                sql_block = get_sql_from_generated(raw)
                if not sql_block:
                    print(f"Failed to generate SQL for {j.path.name}.")
                    j.last_error = "Model returned no SQL."
                    j.schedule_repair_or_fail()
                    continue

                candidate = (
                    f"{j.combined_schema.rstrip()}\n\n"
                    f"{sql_block.strip()}\n"
                )
                ok, err = check_executable_sqlite(candidate)

                if ok:
                    j.on_successful_addition(sql_block)
                    if j.finished and j.path.name not in completed_names:
                        pbar.update(1)
                        completed_names.add(j.path.name)
                else:
                    new_extract = _extract_from_create_table(sql_block.strip())
                    if new_extract:
                        candidate = (
                            f"{j.combined_schema.rstrip()}\n\n"
                            f"{new_extract.strip()}\n"
                        )
                        ok2, err2 = check_executable_sqlite(candidate)
                        if ok2:
                            j.on_successful_addition(new_extract)
                            if j.finished and j.path.name not in completed_names:
                                pbar.update(1)
                                completed_names.add(j.path.name)
                            continue
                    j.last_error = err or "Unknown SQLite error."
                    j.schedule_repair_or_fail()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"⏱️  Total time: {total_time:.2f} seconds for {len(jobs)} files ({len(active)} extended).")
    print("✅ Done.")


# ----------------------------- CLI --------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extend SQL schemas with 15 new tables using the imported LLM adapters in batched rounds.")
    p.add_argument("--folder", type=Path, default=Path("."), help="Folder containing schema files.")
    p.add_argument("--pattern", type=str, default="*.sql", help="Glob pattern to select schema files.")
    p.add_argument("--backend", type=str, default="vllm", choices=["vllm", "openai"], help="Adapter backend to use.")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-14B-FP8", help="Model name (vLLM model path or OpenAI model ID).")
    p.add_argument("--openai-api-key", type=str, default=None, help="API key for OpenAI (the SDK may also read OPENAI_API_KEY env var).")
    p.add_argument("--retries", type=int, default=3, help="Max attempts per batch for a file (including the first).")
    p.add_argument("--out-dir", type=Path, default=None, help="Directory to write outputs (ignored with --in-place).")
    p.add_argument("--in-place", action="store_true", help="Overwrite the input files with the extended schemas.")
    # sampling / generation
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    p.add_argument("--max-tokens", type=int, default=4096, help="Max new tokens to generate per round.")
    p.add_argument("--top-k", type=int, default=None, help="(Compatibility) ignored by adapters; kept to avoid breaking your CLI.")
    p.add_argument("--max-batch-size", type=int, default=16, help="Max number of prompts handed to the model per round.")
    p.add_argument("--tensor_parallel_size", type=int, default=1, help="(vLLM only) Tensor parallel size; ignored by OpenAI adapter.")
    return p.parse_args()


def make_adapter(args: argparse.Namespace) -> ModelAdapter:
    if args.backend == "vllm":
        # The provided VLLMAdapter expects a vLLM model object in its constructor
        # in some implementations; in the shared class it takes an 'Any' model.
        # If your local adapter instead expects a model name/path, construct the
        # vLLM object outside and pass it in.
        #
        # Here we follow the provided signature by creating the vLLM engine and
        # passing it to the adapter if needed. Adjust to your exact adapter init.
        from vllm import LLM  # local import so openai-only users don't need vllm
        engine = LLM(model=args.model, max_model_len=15000, tensor_parallel_size=args.tensor_parallel_size)
        return VLLMAdapter(model=engine)
    else:
        return AsyncOpenAIAdapter(model=args.model, api_key=args.openai_api_key)


def main() -> None:
    args = parse_args()
    adapter = make_adapter(args)

    process_folder_batched(
        folder=args.folder,
        pattern=args.pattern,
        adapter=adapter,
        retries=max(1, args.retries),
        out_dir=args.out_dir,
        in_place=args.in_place,
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        top_k=args.top_k if args.top_k is None else int(args.top_k),
        max_batch_size=max(1, int(args.max_batch_size)),
    )


if __name__ == "__main__":
    main()