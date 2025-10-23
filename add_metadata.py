import argparse
import json
import re
import sys
from typing import List, Dict, Any
import random
from tqdm import tqdm

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
    return len(re.findall(r"[A-Za-z0-9_%/*.+<>!=\\-]+", text))

# -- SQL helpers --
JOIN_RE = re.compile(r"\bjoin\b", re.IGNORECASE)

SQL_COMMANDS = [
    "WITH", "SELECT", "INSERT", "UPDATE", "DELETE",
    "CREATE", "ALTER", "DROP", "TRUNCATE",
    "UNION", "INTERSECT", "EXCEPT",
    "MERGE", "CALL",
]
COMMANDS_RE = re.compile(r"\b(" + "|".join(SQL_COMMANDS) + r")\b", re.IGNORECASE)

# Count CREATE TABLE statements strictly from schema
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

def _split_top_level_commas(s: str):
    parts, buf = [], []
    depth = 0
    in_single = in_double = in_back = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "'" and not in_double and not in_back:
            in_single = not in_single
        elif ch == '"' and not in_single and not in_back:
            in_double = not in_double
        elif ch == '`' and not in_single and not in_double:
            in_back = not in_back
        elif ch == '(' and not (in_single or in_double or in_back):
            depth += 1
        elif ch == ')' and not (in_single or in_double or in_back):
            depth = max(0, depth - 1)

        if ch == ',' and depth == 0 and not (in_single or in_double or in_back):
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1

    if buf:
        parts.append(''.join(buf).strip())
    return parts


def count_columns_from_schema(schema: str) -> int:
    """Count column definitions inside CREATE TABLE statements."""
    if not schema:
        return 0

    pattern = r"""
        CREATE\s+
        (?:TEMP|TEMPORARY\s+)?TABLE\s+
        (?:IF\s+NOT\s+EXISTS\s+)?              # optional IF NOT EXISTS
        (?:                                     # table name (quoted/backticked/bracketed/bare; may be schema-qualified)
           "(?:[^"]+)"
           | `[^`]+`
           | \[[^\]]+\]
           | [\w.]+
        )
        \s*\(
        (.*?)
        \)
        \s*;
    """
    table_defs = re.findall(pattern, schema, flags=re.IGNORECASE | re.DOTALL | re.VERBOSE)

    total_columns = 0
    for cols_block in table_defs:
        parts = _split_top_level_commas(cols_block)
        for col in parts:
            if not col:
                continue
            # Skip table-level constraints
            if re.match(r"^(PRIMARY|FOREIGN|UNIQUE|CHECK|CONSTRAINT|INDEX)\b", col.strip(), re.IGNORECASE):
                continue
            total_columns += 1

    return total_columns

def extend_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    # Map field names
    question = rec.get("prompt", "") or rec.get("question", "") or ""
    query = rec.get("sql_statement", "") or rec.get("query", "") or ""
    schema = rec.get("schema", "") or ""

    # Skip trivial "SELECT 1;"
    if query.strip().upper() == "SELECT 1;":
        return None

    # Token counts split + total
    token_count = {
        "question": count_tokens(question),
        "query": count_tokens(query),
        "schema": count_tokens(schema),
    }
    token_count["total"] = sum(token_count.values())

    num_joins = count_joins(query)
    commands = extract_commands(query)
    num_tables = count_tables_from_schema(schema)
    number_of_columns = count_columns_from_schema(schema)

    # Copy everything except old fields, rename prompt/sql_statement → question/query
    out = {k: v for k, v in rec.items() if k not in ("cmd_type", "prompt", "sql_statement")}
    out["question"] = question
    out["query"] = query
    out["schema"] = schema
    out["token_count"] = token_count
    out["num_joins"] = num_joins
    out["num_tables"] = num_tables
    out["number_of_columns"] = number_of_columns
    return out

def process_stream(instream, outstream):
    results = []
    for line_no, line in tqdm(enumerate(instream, 1), desc="Processing records"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Warning: JSON decode error on line {line_no}: {e}", file=sys.stderr)
            err_obj = {
                "error": "JSONDecodeError",
                "message": str(e),
                "line_no": line_no,
                "raw_line": line,
            }
            results.append(err_obj)
            continue

        try:
            out = extend_record(rec)
            if out is not None:
                results.append(out)
        except Exception as e:
            err_obj = {
                "error": "ProcessingError",
                "message": str(e),
                "line_no": line_no,
                "raw_record": rec,
            }
            results.append(err_obj)

    # 🔀 Shuffle before writing
    random.shuffle(results)

    number_of_skipped = 0
    for obj in tqdm(results, desc="Writing output"):
        if len(obj["question"]) < 20 or len(obj["question"]) > 1024:
            number_of_skipped += 1
            continue
        outstream.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Skipped {number_of_skipped} records with question length < 20 or > 1024 characters.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Extend JSONL SQL records: rename fields, add token counts, JOIN count, command list, table count, and column count; skip trivial SELECT 1 queries."
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
