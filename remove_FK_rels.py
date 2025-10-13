#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
import shutil
from typing import Set

# --- Comment handling ---------------------------------------------------------

def strip_sql_comments(sql_text: str) -> str:
    """Remove /* ... */ block comments and -- line comments."""
    # Remove block comments
    sql_text = re.sub(r"/\*.*?\*/", "", sql_text, flags=re.DOTALL)
    # Remove line comments
    lines = []
    for line in sql_text.splitlines():
        # naive but effective: cut at -- (outside strings is hard; acceptable for constraints removal use-case)
        if "--" in line:
            line = line.split("--", 1)[0]
        lines.append(line)
    return "\n".join(lines)

# --- FK removal routines ------------------------------------------------------

# Inline REFERENCES (within a column definition)
RE_INLINE_REFERENCES = re.compile(
    r"""
    \bREFERENCES\b                                  # REFERENCES
    \s+\w+                                          # referenced table
    (?:\s*\([^)]+\))?                               # optional (col, col2)
    (?:                                             # optional actions/qualifiers
        \s+(?:ON\s+DELETE|ON\s+UPDATE)\s+\w+ |
        \s+MATCH\s+\w+ |
        \s+DEFERRABLE |
        \s+NOT\s+DEFERRABLE |
        \s+INITIALLY\s+\w+
    )*
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Table-level FK constraints inside CREATE TABLE (...) list items
RE_TABLE_LEVEL_FK_WITH_CONSTRAINT = re.compile(
    r"""
    (?:^\s*|,\s*)                                   # start of item or leading comma
    (?:CONSTRAINT\s+\w+\s+)?                        # optional CONSTRAINT name
    FOREIGN\s+KEY\s*\([^)]+\)\s*                    # FOREIGN KEY ( ... )
    REFERENCES\s+\w+(?:\s*\([^)]+\))?               # REFERENCES tbl ( ... )?
    (?:                                             # optional actions
        \s+(?:ON\s+DELETE|ON\s+UPDATE)\s+\w+ |
        \s+MATCH\s+\w+ |
        \s+DEFERRABLE |
        \s+NOT\s+DEFERRABLE |
        \s+INITIALLY\s+\w+
    )*
    (?=                                             # lookahead to not consume next item separator
        \s*,\s* | \s*\)\s* | \s*$                   # next comma, closing paren, or end
    )
    """,
    re.IGNORECASE | re.VERBOSE | re.MULTILINE | re.DOTALL,
)

# ALTER TABLE ... ADD CONSTRAINT ... FOREIGN KEY ... ;
RE_ALTER_ADD_FK_STMT = re.compile(
    r"""
    \bALTER\s+TABLE\b .*?                           # ALTER TABLE ...
    \bADD\s+(?:CONSTRAINT\s+\w+\s+)?                # ADD [CONSTRAINT name]
    FOREIGN\s+KEY\s*\([^)]+\)\s*                    # FOREIGN KEY ( ... )
    REFERENCES\s+\w+(?:\s*\([^)]+\))?               # REFERENCES tbl ( ... )?
    (?:                                             # optional actions
        \s+(?:ON\s+DELETE|ON\s+UPDATE)\s+\w+ |
        \s+MATCH\s+\w+ |
        \s+DEFERRABLE |
        \s+NOT\s+DEFERRABLE |
        \s+INITIALLY\s+\w+
    )*
    \s*;                                            # up to terminating semicolon
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

# After removing items from a CREATE TABLE list, we might leave dangling commas before ')'
RE_FIX_TRAILING_COMMA = re.compile(r",\s*\)")

def remove_foreign_keys_sql(sql_text: str) -> str:
    """
    Remove:
      - ALTER TABLE ... ADD ... FOREIGN KEY ... ;
      - table-level FOREIGN KEY constraints in CREATE TABLE
      - inline REFERENCES ... in column definitions
    Also fixes dangling commas before ')'.
    Comments are stripped to avoid false positives in comments.
    """
    original = sql_text
    # Strip comments for reliable matching; operate on cleaned text (comments won't be preserved)
    cleaned = strip_sql_comments(original)

    # Remove ALTER TABLE FK statements entirely
    cleaned = RE_ALTER_ADD_FK_STMT.sub("", cleaned)

    # Remove table-level FKs inside CREATE TABLE blocks
    cleaned = RE_TABLE_LEVEL_FK_WITH_CONSTRAINT.sub("", cleaned)

    # Remove inline REFERENCES on columns
    cleaned = RE_INLINE_REFERENCES.sub("", cleaned)

    # Fix dangling commas like ", )"
    cleaned = RE_FIX_TRAILING_COMMA.sub(")", cleaned)

    # Clean up repeated commas/spaces and empty lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return cleaned

# --- CSV handling & file targeting -------------------------------------------

def load_marked_prefixes(csv_path: Path) -> Set[str]:
    """
    Read CSV (file_name,foreign_key) and return a set of basenames (without .sql)
    for rows where foreign_key is true/True/1/yes.
    """
    true_vals = {"true", "1", "yes", "y"}
    prefixes: Set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "file_name" not in reader.fieldnames or "foreign_key" not in reader.fieldnames:
            raise SystemExit("CSV must have headers: file_name, foreign_key")
        for row in reader:
            fname = (row.get("file_name") or "").strip()
            fkstr = (row.get("foreign_key") or "").strip().lower()
            if not fname:
                continue
            stem = Path(fname).stem  # remove .sql
            if fkstr in true_vals:
                prefixes.add(stem)
    return prefixes

def should_process(file_path: Path, prefixes: Set[str]) -> bool:
    """Return True if file's stem starts with any marked prefix."""
    stem = file_path.stem
    return any(stem.startswith(pref) for pref in prefixes)

# --- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Remove foreign key references from .sql files whose names start with any prefix marked true in the CSV."
    )
    ap.add_argument("csv", type=Path, help="CSV with headers: file_name,foreign_key")
    ap.add_argument("folder", type=Path, help="Root folder to scan for .sql files")
    ap.add_argument("--ext", default=".sql", help="Extension to scan (default: .sql)")
    ap.add_argument("--apply", action="store_true", help="Actually modify files (default: dry run)")
    ap.add_argument("--backup", action="store_true", help="Create a .bak copy next to each modified file")
    ap.add_argument("--encoding", default="utf-8", help="File encoding for reading/writing (default: utf-8)")
    args = ap.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")
    if not args.folder.is_dir():
        raise SystemExit(f"Folder not found: {args.folder}")

    prefixes = load_marked_prefixes(args.csv)
    if not prefixes:
        print("No files marked as having foreign keys (true) in CSV; nothing to do.")
        return

    sql_files = sorted(p for p in args.folder.rglob(f"*{args.ext}") if p.is_file())
    targets = [p for p in sql_files if should_process(p, prefixes)]

    print(f"Marked prefixes: {sorted(prefixes)}")
    print(f"SQL files scanned: {len(sql_files)}")
    print(f"Target files to sanitize: {len(targets)}")
    if not targets:
        return

    changed = 0
    for path in targets:
        try:
            text = path.read_text(encoding=args.encoding, errors="ignore")
        except Exception:
            text = path.read_text(encoding="latin-1", errors="ignore")

        new_text = remove_foreign_keys_sql(text)

        if new_text != text:
            changed += 1
            if args.apply:
                if args.backup:
                    bak = path.with_suffix(path.suffix + ".bak")
                    try:
                        shutil.copy2(path, bak)
                    except Exception:
                        # fallback simple copy
                        shutil.copy(path, bak)
                path.write_text(new_text, encoding=args.encoding)
                print(f"[MODIFIED] {path}")
            else:
                print(f"[WOULD MODIFY] {path}")
        else:
            print(f"[UNCHANGED] {path} (no FK patterns found)")

    print("\n=== Summary ===")
    print(f"Targets: {len(targets)}")
    print(f"Changed: {changed}")
    print(f"Dry run: {not args.apply}")
    if args.backup and args.apply:
        print("Backups: *.sql.bak created next to modified files")

if __name__ == "__main__":
    main()
