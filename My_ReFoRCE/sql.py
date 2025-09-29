from __future__ import annotations

import re
from typing import List, Sequence, Tuple, Optional
from collections import defaultdict

from My_ReFoRCE.in_memory_db import InMemoryDB
from My_ReFoRCE.model import ModelAdapter, GenerationConfig

_WS = re.compile(r"\s+")
_TRL_SEMI = re.compile(r";\s*$")

def _normalize_sql(sql: str) -> str:
    s = sql.strip()
    s = _TRL_SEMI.sub("", s)
    return _WS.sub(" ", s).lower()

def _ensure_semicolon(sql: str) -> str:
    s = sql.strip()
    return s if s.endswith(";") else s + ";"

def _compress_schema(db: InMemoryDB, max_cols: int = 8) -> str:
    lines = []
    for t in db.table_names():
        cols = db.columns(t)
        prev = ", ".join(f"{c}:{typ}" for c, typ in cols[:max_cols])
        if len(cols) > max_cols:
            prev += ", ..."
        lines.append(f"{t}({prev})")
    return "\n".join(lines)

def _strip_md_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[\w-]*\n?", "", s).strip()
        s = re.sub(r"\n?```$", "", s).strip()
    return s

def _gen_system_prompt() -> str:
    return (
        "You are a Text-to-SQL generator for SQLite.\n"
        "Rules:\n"
        " - Output ONLY SQL (one statement per candidate).\n"
        " - If you produce multiple candidates, separate them with a line that is exactly ###.\n"
        " - No explanations, no markdown."
    )

def _vote_system_prompt() -> str:
    return (
        "You are a SQL judge. You will receive several SQL candidates for the SAME task and schema.\n"
        "Pick the SINGLE best candidate that is syntactically valid SQLite and most likely to answer the task.\n"
        "Output EXACTLY the chosen SQL statement, with no edits, no commentary, no markdown."
    )

def _build_generation_prompt(task: str, compressed_schema: str) -> str:
    return (
        f"Task: {task}\n\n"
        "Requirements:\n"
        "- Dialect: SQLite\n"
        "- Output 1–4 alternative SQL candidates if unsure (use ### separators).\n"
        "- ONE statement per candidate.\n"
        "- Use ONLY the given schema; do not invent tables/columns.\n\n"
        f"Schema:\n{compressed_schema}\n"
    )

def _build_vote_prompt(task: str, compressed_schema: str, candidates: List[str]) -> str:
    labeled = []
    for i, c in enumerate(candidates, 1):
        labeled.append(f"<<CANDIDATE {i}>>\n{c.strip()}\n<<END>>")
    cand_block = "\n".join(labeled)
    return (
        f"Task: {task}\n\n"
        f"Schema:\n{compressed_schema}\n\n"
        "Candidates:\n"
        f"{cand_block}\n\n"
        "Return ONLY the single best candidate SQL, copied verbatim."
    )

def text2sql(
    items: Sequence[Tuple[str, str]],
    adapter: ModelAdapter,
    cfg: Optional[GenerationConfig] = None,
    max_votes_per_item: int = 4,
) -> List[str]:
    """
    Batch text-to-SQL with batch generation and batch voting.

    Args:
        items: sequence of (prompt, schema_ddl_string)
        adapter: a ModelAdapter (e.g., VLLMAdapter)
        cfg: GenerationConfig; defaults are fine
        max_votes_per_item: target # of alternative candidates per item (LLM may return fewer)

    Returns:
        List[str]: one SQL statement per input item.
    """
    if cfg is None:
        cfg = GenerationConfig(temperature=1.0, top_p=0.95, max_tokens=512)

    # 1) Build compressed schemas (and keep DBs around only for schema introspection)
    compressed_schemas: List[str] = []
    for _, ddl in items:
        if not isinstance(ddl, str):
            raise TypeError("Each schema must be a DDL string with CREATE TABLE statements")
        db = InMemoryDB(ddl)
        try:
            compressed_schemas.append(_compress_schema(db))
        finally:
            db.close()

    # 2) Batch GENERATION: one prompt per item
    gen_prompts: List[str] = []
    for (task, _), comp in zip(items, compressed_schemas):
        gen_prompts.append(_build_generation_prompt(task, comp))

    gen_outputs = adapter.batch_generate(
        prompts=gen_prompts,
        system_prompt=_gen_system_prompt(),
        cfg=cfg,
    )
    # gen_outputs shape: [len(items)][num_candidates_as_big_strings_split_by_adapter]

    # Normalize & limit candidates per item
    all_candidates: List[List[str]] = []
    for cand_list in gen_outputs:
        flat: List[str] = []
        for c in cand_list:
            s = _strip_md_fences(c)
            # Split on ### if adapter didn't already do so thoroughly
            parts = [p.strip() for p in s.split("###") if p.strip()]
            flat.extend(parts)
        # keep a modest number per item
        unique = []
        seen = set()
        for s in flat:
            key = _normalize_sql(s)
            if key not in seen:
                seen.add(key)
                unique.append(s)
            if len(unique) >= max_votes_per_item:
                break
        if not unique:
            unique = ["SELECT 1;"]
        all_candidates.append(unique)

    # 3) Batch VOTING: one vote prompt per item
    vote_prompts: List[str] = []
    for (task, _), comp, cands in zip(items, compressed_schemas, all_candidates):
        vote_prompts.append(_build_vote_prompt(task, comp, cands))

    vote_outputs = adapter.batch_generate(
        prompts=vote_prompts,
        system_prompt=_vote_system_prompt(),
        cfg=GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=256),
    )

    # 4) Parse winners, with fallbacks to majority vote locally if needed
    winners: List[str] = []
    for cands, judge_outs in zip(all_candidates, vote_outputs):
        # Prefer the judge's first output
        chosen_raw = _strip_md_fences(judge_outs[0] if judge_outs else "").strip()

        # If the judge returned exactly one of the candidates (normalized), use it
        norm_to_orig = { _normalize_sql(c): c for c in cands }
        norm_choice = _normalize_sql(chosen_raw)
        if norm_choice in norm_to_orig:
            winners.append(_ensure_semicolon(norm_to_orig[norm_choice]))
            continue

        # Else try to find the closest exact-text match ignoring trailing semicolon
        for c in cands:
            if chosen_raw.rstrip("; \n\t") == c.rstrip("; \n\t"):
                winners.append(_ensure_semicolon(c))
                break
        else:
            # Final local fallback: pick the first candidate
            winners.append(_ensure_semicolon(cands[0]))

    return winners
