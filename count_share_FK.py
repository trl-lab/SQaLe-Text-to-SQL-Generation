#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from collections import Counter

def strip_sql_comments(sql_text: str) -> str:
    """
    Remove SQL comments:
      - /* ... */ block comments
      - -- ... line comments
    (Basic string literal handling: avoids cutting inside quotes)
    """
    # Remove block comments
    no_block = re.sub(r"/\*.*?\*/", "", sql_text, flags=re.DOTALL)

    out_lines = []
    for line in no_block.splitlines():
        cut_at = None
        in_single = False
        in_double = False
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif ch == "-" and not in_single and not in_double:
                if i + 1 < len(line) and line[i + 1] == "-":
                    cut_at = i
                    break
            i += 1
        out_lines.append(line if cut_at is None else line[:cut_at])
    return "\n".join(out_lines)

# Detect ANY indication of foreign-key relationships:
# - Table-level: "FOREIGN KEY" (optionally with CONSTRAINT ...)
# - Inline column-level: "... REFERENCES other_table(col)"
FK_PATTERNS = [
    re.compile(r"\bFOREIGN\s+KEY\b", re.IGNORECASE),
    re.compile(r"\bREFERENCES\b", re.IGNORECASE),
]

def file_has_foreign_keys(clean_sql: str) -> bool:
    for pat in FK_PATTERNS:
        if pat.search(clean_sql):
            return True
    return False

def main():
    ap = argparse.ArgumentParser(
        description="Count SQL files without any foreign key relations and report their percentage share."
    )
    ap.add_argument("folder", type=Path, help="Folder containing .sql files")
    ap.add_argument("--ext", default=".sql", help="File extension to include (default: .sql)")
    ap.add_argument("--report", type=Path, default=None,
                    help="Optional CSV report listing file and has_foreign_keys (true/false)")
    args = ap.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"Folder not found: {args.folder}")

    files = sorted(p for p in args.folder.rglob(f"*{args.ext}") if p.is_file())
    if not files:
        raise SystemExit(f"No files with extension '{args.ext}' found under {args.folder}")

    results = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = p.read_text(encoding="latin-1", errors="ignore")
        cleaned = strip_sql_comments(text)
        has_fk = file_has_foreign_keys(cleaned)
        results.append((p, has_fk))

    total = len(results)
    no_fk_count = sum(1 for _, has_fk in results if not has_fk)
    share = (no_fk_count / total) * 100.0

    # Quick summary
    buckets = Counter("has_fk" if has_fk else "no_fk" for _, has_fk in results)
    print("=== Foreign-Key Presence Summary ===")
    print(f"Total files scanned: {total}")
    print(f"Files with NO foreign keys: {no_fk_count}")
    print(f"Files WITH foreign keys:   {buckets['has_fk']}")
    print(f"Percentage with NO foreign keys: {share:.2f}%")

    if args.report:
        import csv
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "has_foreign_keys"])
            for p, has_fk in results:
                w.writerow([str(p), str(has_fk).lower()])
        print(f"\nReport written to: {args.report}")

if __name__ == "__main__":
    main()
