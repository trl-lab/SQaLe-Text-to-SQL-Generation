from __future__ import annotations

import re
from typing import List, Sequence, Tuple, Optional
import time

from My_ReFoRCE.in_memory_db import InMemoryDB
from My_ReFoRCE.model import ModelAdapter, GenerationConfig
from My_ReFoRCE.utils import split_sql_statements  # NEW: to verify single-statement

# ----------------------------
# Helpers
# ----------------------------
_WS = re.compile(r"\s+")
_TRL_SEMI = re.compile(r";\s*$")
SQL_FENCE_RE = re.compile(r"```sql\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
ANY_FENCE_RE = re.compile(r"```+\s*(.*?)\s*```+", re.DOTALL)

def _normalize_sql(sql: str) -> str:
    s = sql.strip()
    s = _TRL_SEMI.sub("", s)
    return _WS.sub(" ", s).lower()

def _ensure_semicolon(sql: str) -> str:
    s = sql.strip()
    return s if s.endswith(";") else s + ";"

def _extract_sql_from_fence(text: str) -> Optional[str]:
    # Try to find all SQL fenced blocks and return the last one
    matches = list(SQL_FENCE_RE.finditer(text))
    if matches:
        last = matches[-1]
        if last.group(1).strip():
            return last.group(1).strip()
    # Try to find all generic fenced blocks and return the last one
    matches2 = list(ANY_FENCE_RE.finditer(text))
    if matches2:
        last2 = matches2[-1]
        if last2.group(1).strip():
            return last2.group(1).strip()

    # Try to find just the opening ```sql without closing ```
    sql_start = re.search(r"```sql\s*", text, re.IGNORECASE)
    if sql_start:
        start_idx = sql_start.end()
        remaining_text = text[start_idx:]
        semi_idx = remaining_text.find(";")
        if semi_idx != -1:
            return remaining_text[: semi_idx + 1].strip()
        else:
            return remaining_text.strip()

    # As a last resort, try to find the last colon before the last semicolon
    semi_idx = text.rfind(";")
    if semi_idx != -1:
        prev_colon = text.rfind(":", 0, semi_idx)
        if prev_colon != -1 and prev_colon < semi_idx:
            candidate = text[prev_colon + 1 : semi_idx + 1].strip()
            if candidate:
                return candidate

    return None

def _compress_schema(db: InMemoryDB, max_cols: int = 8) -> str:
    lines = []
    for t in db.table_names():
        cols = db.columns(t)
        prev = ", ".join(f"{c}:{typ}" for c, typ in cols[:max_cols])
        if len(cols) > max_cols:
            prev += ", ..."
        lines.append(f"{t}({prev})")
    return "\n".join(lines)

def _is_single_statement(sql: str) -> bool:
    stmts = [s for s in split_sql_statements(sql) if s.strip()]
    return len(stmts) == 1

def _is_executable(sql: str, ddl: str) -> bool:
    """
    Builds a fresh in-memory DB from the provided DDL and tries to execute the SQL.
    Returns True iff it executes without error.
    """
    db = InMemoryDB(ddl)
    try:
        ok, _err = db.try_exec(sql)
        return bool(ok)
    finally:
        db.close()

# ----------------------------
# Prompts
# ----------------------------
def _gen_system_single() -> str:
    return (
        "You are a Text-to-SQL generator for SQLite.\n"
        "Return ONE SQL statement inside a fenced code block:\n"
        "```sql\n<your single SQL here>\n```\n"
    )

def _build_generation_prompt(task: str, compressed_schema: str) -> str:
    return (
        f"Generate the SQLite statement for the following questions: {task}\n\n"
        "Constraints:\n"
        "- Dialect: SQLite\n"
        "- Use ONLY the given schema; do not invent tables/columns\n"
        "- Return exactly ONE valid SQL statement that accomplishes the task\n"
        "- Use JOINs if needed. In general try to write easy to read SQL.\n"
        "- If needed, use subqueries or CTEs (WITH ...) to simplify complex queries for the user.\n"
        "- Put the SQL inside a ```sql\n<your single SQL here>\n``` fenced block\n\n"
        f"This is the SQLite schema:\n{compressed_schema}\n"
    )

def _vote_system() -> str:
    return (
        "You are a SQL judge. You will receive several SQL candidates for the SAME task and schema.\n"
        "Pick ONE of the provided candidates and return it INSIDE a ```sql fenced block, copied verbatim.\n"
        "No commentary."
    )

def _build_vote_prompt(task: str, compressed_schema: str, candidates: List[str]) -> str:
    labeled = []
    for i, c in enumerate(candidates, 1):
        labeled.append(f"<<CANDIDATE {i}>>\n```sql\n{c.strip()}\n```\n<<END>>")
    cand_block = "\n".join(labeled)
    return (
        f"Task: {task}\n\n"
        f"Schema:\n{compressed_schema}\n\n"
        "Candidates:\n"
        f"{cand_block}\n\n"
        "Return ONLY the single best candidate inside a ```sql fenced block."
    )

# ----------------------------
# Public API
# ----------------------------
def text2sql(
    items: Sequence[Tuple[str, str]],
    adapter: ModelAdapter,
    *,
    cfg_generate: Optional[GenerationConfig] = None,
    cfg_vote: Optional[GenerationConfig] = None,
    candidates_per_item: int = 2,
    num_trials: int = 3,
) -> List[str]:
    """
    Batch Text-to-SQL:
      - For each (prompt, schema): call the model multiple times; each call produces ONE candidate in ```sql``` block.
      - Each candidate is validated by executing it in an InMemoryDB built from the provided schema.
      - Then do a single batch voting pass (one prompt per item) where the judge returns ONE winner in ```sql``` block.
      - Returns one SQL string per input (with trailing semicolon ensured).

    Args:
        items: sequence of (prompt, schema_ddl_string)
        adapter: ModelAdapter (e.g., VLLMAdapter)
        cfg_generate: GenerationConfig for candidate generation (defaults are fine)
        cfg_vote: GenerationConfig for judge (defaults deterministic)
        candidates_per_item: number of single-candidate generations per item
    """
    if cfg_generate is None:
        cfg_generate = GenerationConfig(temperature=1.0, top_p=0.95, max_tokens=4096)
    if cfg_vote is None:
        cfg_vote = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=4096)

    # 1) Prepare compressed schemas
    compressed_schemas: List[str] = []
    for _, ddl in items:
        if not isinstance(ddl, str):
            raise TypeError("Each schema must be a DDL string with CREATE TABLE statements")
        db = InMemoryDB(ddl)
        try:
            compressed_schemas.append(_compress_schema(db))
        finally:
            db.close()

    # 2) MULTI-CALL GENERATION: repeat batch calls; each call yields ONE candidate per item
    # Build the generation prompts once
    gen_prompts = [
        _build_generation_prompt(task, comp) for (task, _), comp in zip(items, compressed_schemas)
    ]

    per_item_candidates: List[List[str]] = [[] for _ in items]
    seen_norms_per_item: List[set] = [set() for _ in items]

    print("Starting candidate generation...")
    current_time = time.time()

    for _ in range(max(1, candidates_per_item)):
        outs = adapter.batch_generate(
            prompts=gen_prompts,
            system_prompt=_gen_system_single(),
            cfg=cfg_generate
        )  # shape: [len(items)][num_candidates_from_adapter]

        # Extract exactly one candidate from this round per item (first valid fenced block found)
        for idx, raw in enumerate(outs):
            # Concatenate model outputs (some adapters may still split); take the first that parses
            extracted: Optional[str] = None
            try:
                extracted = _extract_sql_from_fence(raw)

                if not extracted:
                    raise ValueError("Extraction failed: No SQL extracted from model output.")

                # Ensure trailing semicolon for consistent splitting/execution
                extracted = _ensure_semicolon(extracted)

                # Enforce single-statement rule BEFORE dedup/execution
                if not _is_single_statement(extracted):
                    raise ValueError("Validation failed: Not a single SQL statement.")

                norm = _normalize_sql(extracted)
                if norm in seen_norms_per_item[idx]:
                    raise ValueError("Deduplication failed: Duplicate candidate.")

                # ---- NEW: executability check against the actual schema ----
                _task, ddl = items[idx]
                if not _is_executable(extracted, ddl):
                    # not executable -> discard this candidate
                    raise ValueError("Executability failed: SQL not executable against schema.")

                # Passed all checks: record it
                seen_norms_per_item[idx].add(norm)
                per_item_candidates[idx].append(extracted)
            except Exception as e:
                # extraction/validation failed -> skip this candidate
                # print(f"Candidate {idx} skipped: {e}")
                last_error = f"{e} with sql `{extracted}`"
                continue

    print("Candidate generation complete.")
    print(f"Time taken: {time.time() - current_time:.2f} seconds")

    print("\nStarting voting...")
    current_time = time.time()

    # Ensure at least one candidate per item
    for i, cands in enumerate(per_item_candidates):
        if not cands:
            # last resort fallback: a trivially executable statement
            per_item_candidates[i] = ["SELECT 1;"]

    # 3) SINGLE BATCH VOTING (judge returns one fenced SQL per item)
    vote_prompts = []
    for (task, _), comp, cands in zip(items, compressed_schemas, per_item_candidates):
        vote_prompts.append(_build_vote_prompt(task, comp, cands))

    judge_outs = adapter.batch_generate(
        prompts=vote_prompts,
        system_prompt=_vote_system(),
        cfg=cfg_vote
    )

    print("Voting complete.")
    print(f"Time taken: {time.time() - current_time:.2f} seconds")

    # 4) Parse winners; map back to exact candidate when possible
    winners: List[str] = []
    for cands, outs in zip(per_item_candidates, judge_outs):
        chosen = None
        extracted = None
        # get first judged output, extract sql fence
        if outs:
            extracted = _extract_sql_from_fence(outs[0])
        if extracted:
            norm_choice = _normalize_sql(extracted)
            # exact normalized match among candidates?
            norm_map = {_normalize_sql(c): c for c in cands}
            if norm_choice in norm_map:
                chosen = norm_map[norm_choice]
            else:
                # try loose match ignoring trailing semicolon
                for c in cands:
                    if extracted.rstrip(";\n\t ") == c.rstrip(";\n\t "):
                        chosen = c
                        break
        if chosen is None:
            chosen = cands[0]
        winners.append(_ensure_semicolon(chosen))

    return winners
