#!/usr/bin/env python3
import re
import csv
from pathlib import Path
import argparse

def strip_sql_comments(sql_text: str) -> str:
    """Remove SQL block and line comments (/*...*/, -- ...)."""
    sql_text = re.sub(r"/\*.*?\*/", "", sql_text, flags=re.DOTALL)
    lines = []
    for line in sql_text.splitlines():
        cut = line.split("--", 1)[0]
        lines.append(cut)
    return "\n".join(lines)

def has_foreign_key(sql_text: str) -> bool:
    """Return True if the SQL text contains a foreign key reference."""
    cleaned = strip_sql_comments(sql_text)
    return bool(
        re.search(r"\bFOREIGN\s+KEY\b", cleaned, re.IGNORECASE)
        or re.search(r"\bREFERENCES\b", cleaned, re.IGNORECASE)
    )

def main():
    parser = argparse.ArgumentParser(
        description="Scan SQL files and store CSV list of whether each file has a foreign key relation."
    )
    parser.add_argument("folder", type=Path, help="Folder containing .sql files")
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("foreign_keys.csv"),
        help="Output CSV file (default: foreign_keys.csv)"
    )
    parser.add_argument("--ext", default=".sql", help="File extension to scan (default: .sql)")
    args = parser.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"Error: folder not found → {args.folder}")

    files = sorted(args.folder.rglob(f"*{args.ext}"))
    if not files:
        raise SystemExit(f"No {args.ext} files found in {args.folder}")

    rows = []
    for file in files:
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = file.read_text(encoding="latin-1", errors="ignore")
        fk = has_foreign_key(text)
        rows.append((file.name, fk))

    # Write CSV
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "foreign_key"])
        writer.writerows(rows)

    # Summary
    total = len(rows)
    fk_true = sum(1 for _, fk in rows if fk)
    fk_false = total - fk_true
    print(f"Wrote {total} records to {args.output}")
    print(f"Files with foreign keys: {fk_true}")
    print(f"Files without foreign keys: {fk_false} ({fk_false/total*100:.2f}%)")

if __name__ == "__main__":
    main()
