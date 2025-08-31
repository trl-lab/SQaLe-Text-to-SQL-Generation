#!/usr/bin/env python3
"""
Generate a per-file data dictionary from folders of SQL files (each file contains only CREATE TABLE statements).
For each file, we send the *entire file* to the LLM and expect a JSON array of table descriptions.

Usage:
  python gen_dd_from_sql_per_file.py --input ./sql --output ./data_dictionary.json --model gemma3
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from ollama import chat
from ollama import ChatResponse
from tqdm import tqdm


def make_prompt_for_file(sql_text: str) -> List[Dict[str, str]]:
    system = {
        "role": "system",
        "content": (
            "You are a precise data dictionary writer. "
            "Given one SQL file that contains only CREATE TABLE statements, "
            "produce concise, accurate plain-English descriptions for every table in the file."
        ),
    }
    user = {
        "role": "user",
        "content": f"""
            Write ONLY JSON (no markdown, no comments, no extra text). The JSON must be an array where each element matches:

            {{
            "table_name": "string (exact table name)",
            "friendly_name": "short readable name",
            "description": "2-3 sentence summary of what the table represents and how it's used",
            "notes": "optional clarifying notes if any (or empty string)"
            }}

            Rules:
            - Output a *single JSON array* with one object per table found in the input.
            - Be brief and factual. If uncertain, say "unknown" instead of inventing details.
            - Preserve exact table and column names.
            - Do NOT wrap in code fences.

            SQL file contents:
            {sql_text}
        """.strip(),
    }
    return [system, user]


def parse_llm_json(s: str) -> Any:
    s = s.strip()
    # Be tolerant of accidental code fences or leading/trailing junk
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    # Try to isolate the first top-level JSON array if extra text slipped in
    first_bracket = s.find('[')
    last_bracket = s.rfind(']')
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        candidate = s[first_bracket:last_bracket+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return json.loads(s)


def normalize_table_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure required fields and sensible fallbacks
    tn = obj.get("table_name", "").strip() if isinstance(obj.get("table_name"), str) else ""
    obj.setdefault("table_name", tn)
    obj.setdefault("friendly_name", (tn or "unknown").replace("_", " ").title())
    obj.setdefault("description", "")
    return obj


def describe_file_with_llm(model: str, sql_text: str) -> Dict[str, Any]:
    messages = make_prompt_for_file(sql_text)
    response: ChatResponse = chat(model=model, messages=messages)
    raw = response["message"]["content"]

    try:
        data = parse_llm_json(raw)

        # Accept either an array or {"tables":[...]} (some models do this)
        if isinstance(data, dict) and "tables" in data and isinstance(data["tables"], list):
            table_list = data["tables"]
        elif isinstance(data, list):
            table_list = data
        else:
            raise ValueError("LLM did not return a JSON array or an object with 'tables'.")

        # Normalize each element
        tables = []
        for item in table_list:
            if not isinstance(item, dict):
                continue
            tables.append(normalize_table_obj(item))

        return {"tables": tables, "notes": ""}

    except Exception as e:
        # Capture raw output for debugging; keep pipeline moving
        return {
            "tables": [],
            "notes": f"LLM parse error; captured raw output.\nError: {e}\nRaw:\n{raw}"
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to folder containing .sql files (each contains only CREATE TABLE statements)")
    parser.add_argument("--output", required=True, help="Path to write the consolidated JSON file")
    parser.add_argument("--model", default="qwen3:30b", help="Ollama model name (e.g., gemma3)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise SystemExit(f"Input path is not a directory: {input_dir}")

    sql_files = sorted([p for p in input_dir.rglob("*.sql") if p.is_file()])
    if not sql_files:
        raise SystemExit(f"No .sql files found under: {input_dir}")

    files_out: List[Dict[str, Any]] = []
    total_tables = 0

    try:
        for sql_path in tqdm(sql_files, desc="Processing SQL files"):
            sql_text = sql_path.read_text(encoding="utf-8", errors="ignore")
            # Send the whole file at once (assumed to contain only CREATE TABLE statements)
            result = describe_file_with_llm(args.model, sql_text)
            files_out.append({
                "source_file": str(sql_path),
                "tables": result.get("tables", []),
                "notes": result.get("notes", "")
            })
            total_tables += len(result.get("tables", []))
    except Exception as e:
        print(f"Error processing SQL file {sql_path}: {e}")
        pass

    payload = {
        "model": args.model,
        "files": files_out,
        "stats": {
            "files_scanned": len(sql_files),
            "total_tables_described": total_tables
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {out_path} with {len(sql_files)} files and {total_tables} table descriptions.")

if __name__ == "__main__":
    main()
