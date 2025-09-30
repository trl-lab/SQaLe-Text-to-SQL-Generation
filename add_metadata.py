import argparse
import json
import re
import sys
from typing import List, Dict, Any

# -- Optional: tiktoken for more realistic token counts (if installed) --
def _load_tiktoken():
    try:
        import tiktoken  # type: ignore
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

_TIKTOKEN = _load_tiktoken()

def count_tokens(text: str) -> int:
    """Count tokens for a text. Prefer tiktoken if available; otherwise use a heuristic."""
    if not text:
        return 0
    if _TIKTOKEN is not None:
        try:
            return len(_TIKTOKEN.encode(text))
        except Exception:
            pass
    return len(re.findall(r"[A-Za-z0-9_%/*.+\-<>!=]+", text))

# -- SQL helpers --
JOIN_RE = re.compile(r"\bjoin\b", re.IGNORECASE)

SQL_COMMANDS = [
    "WITH", "SELECT", "INSERT", "UPDATE", "DELETE",
    "CREATE", "ALTER", "DROP", "TRUNCATE",
    "UNION", "INTERSECT", "EXCEPT",
    "MERGE", "CALL", "JOIN", "WHERE", "GROUP BY", "HAVING", "ORDER BY",
    "LIMIT", "OFFSET", "VALUES", "SET", "RETURNING", "INTO", "ON", "AS", "AND", "OR", "NOT"
]
COMMANDS_RE = re.compile(r"\b(" + "|".join(SQL_COMMANDS) + r")\b", re.IGNORECASE)

# Count CREATE TABLE statements strictly from schema (case-insensitive).
# Matches: CREATE TABLE, CREATE TEMP TABLE, CREATE TABLE IF NOT EXISTS, etc.
CREATE_TABLE_RE = re.compile(r"\bCREATE\s+(?:TEMP|TEMPORARY\s+)?TABLE\b", re.IGNORECASE)

def extract_commands(sql: str) -> List[str]:
    if not sql:
        return []
    seen = set()
    ordered: List[str] = []
    for m in COMMANDS_RE.finditer(sql):
        cmd = m.group(1).upper()
        if cmd not in seen:
            seen.add(cmd)
            ordered.append(cmd)
    return ordered

def count_joins(sql: str) -> int:
    if not sql:
        return 0
    return len(JOIN_RE.findall(sql))

def count_tables_from_schema(schema: str) -> int:
    if not schema:
        return 0
    return len(CREATE_TABLE_RE.findall(schema))

def extend_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    prompt = rec.get("prompt", "") or ""
    sql = rec.get("sql_statement", "") or ""
    schema = rec.get("schema", "") or ""

    combined = f"{schema}\n{prompt}\n{sql}"
    token_count = count_tokens(combined)
    num_joins = count_joins(sql)
    commands = extract_commands(sql)
    num_tables = count_tables_from_schema(schema)

    out = {k: v for k, v in rec.items() if k != "cmd_type"}
    out["token_count"] = token_count
    out["num_joins"] = num_joins
    out["commands"] = commands
    out["num_tables"] = num_tables
    return out

def process_stream(instream, outstream):
    for line_no, line in enumerate(instream, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            err_obj = {
                "error": "JSONDecodeError",
                "message": str(e),
                "line_no": line_no,
                "raw_line": line,
            }
            outstream.write(json.dumps(err_obj, ensure_ascii=False) + "\n")
            continue

        try:
            out = extend_record(rec)
            outstream.write(json.dumps(out, ensure_ascii=False) + "\n")
        except Exception as e:
            err_obj = {
                "error": "ProcessingError",
                "message": str(e),
                "line_no": line_no,
                "raw_record": rec,
            }
            outstream.write(json.dumps(err_obj, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Extend JSONL SQL records with token counts, JOIN count, command list, and number of tables from schema; remove cmd_type."
    )
    parser.add_argument("--input", help="Path to input JSONL file")
    parser.add_argument("-o", "--output", help="Path to output JSONL file (default: stdout)")
    args = parser.parse_args()

    instream = open(args.input, "r", encoding="utf-8")
    if args.output:
        outstream = open(args.output, "w", encoding="utf-8")
        close_out = True
    else:
        outstream = sys.stdout
        close_out = False

    try:
        process_stream(instream, outstream)
    finally:
        instream.close()
        if close_out:
            outstream.close()

if __name__ == "__main__":
    main()
