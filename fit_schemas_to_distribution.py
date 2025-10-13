#!/usr/bin/env python3
import argparse
import os
import re
import random
import shutil
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np


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
    y = y / y.sum()
    return y


def strip_sql_comments(sql_text: str) -> str:
    """
    Remove SQL comments:
      - block comments: /* ... */
      - line comments: -- ... (to end of line)
    Keeps string literals intact (basic handling).
    """
    # Remove block comments
    no_block = re.sub(r"/\*.*?\*/", "", sql_text, flags=re.DOTALL)

    # Remove line comments (but not inside quoted strings)
    out_lines = []
    for line in no_block.splitlines():
        # naive approach: cut at -- if it's not inside single/double quotes
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


CREATE_TABLE_RE = re.compile(r"\bCREATE\s+TABLE\b", re.IGNORECASE)


def count_create_tables(sql_text: str) -> int:
    cleaned = strip_sql_comments(sql_text)
    return len(CREATE_TABLE_RE.findall(cleaned))


def main():
    parser = argparse.ArgumentParser(
        description="Count CREATE TABLEs in .sql files and semi-randomly delete files to match a target distribution."
    )
    parser.add_argument("folder", type=Path, help="Folder containing .sql files")
    parser.add_argument(
        "--ext", default=".sql", help="File extension to consider (default: .sql)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=None,
        help="Mode (peak) of target distribution in 'table count' units. Default = round(0.4 * max_count).",
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=None,
        help="Maximum x for distribution (inclusive). Default = observed max table count.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete/move files. Without this flag, performs a dry run.",
    )
    parser.add_argument(
        "--trash-dir",
        type=Path,
        default=None,
        help="If provided, move files here instead of deleting. The directory will be created if it doesn't exist.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write a CSV report of counts and decisions.",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    folder = args.folder
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    files = sorted(p for p in folder.rglob(f"*{args.ext}") if p.is_file())
    if not files:
        raise SystemExit(f"No files with extension '{args.ext}' found in {folder}")

    # Count tables per file
    file_counts = {}
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            # Retry with latin-1 fallback
            text = p.read_text(encoding="latin-1", errors="ignore")
        file_counts[p] = count_create_tables(text)

    # Histogram of observed counts
    counts = list(file_counts.values())
    total_files = len(counts)
    observed_max = max(counts) if counts else 0
    observed_hist = Counter(counts)

    # Target distribution
    max_val = args.max_val if args.max_val is not None else observed_max
    if max_val < observed_max:
        print(
            f"Warning: --max-val ({max_val}) < observed max ({observed_max}); adjusting to observed max."
        )
        max_val = observed_max

    mode = args.mode if args.mode is not None else max(1, round(0.4 * max_val))
    target_probs = get_probability_distribution(mode, max_val)

    # Target counts per bucket (rounded). We'll cap by how many we actually have.
    target_counts = {k: int(round(target_probs[k] * total_files)) for k in range(max_val + 1)}

    # Build bucket -> list of files
    buckets = defaultdict(list)
    for p, c in file_counts.items():
        buckets[c].append(p)

    # Decide which files to KEEP per bucket
    keep = set()
    delete = set()
    decisions = []  # (path, count, action)

    for k in range(max_val + 1):
        bucket_files = buckets.get(k, [])
        n_have = len(bucket_files)
        n_target = target_counts.get(k, 0)
        # We can only keep up to what's available
        n_keep = min(n_have, n_target)

        # semi-random selection (stable with seed)
        random.shuffle(bucket_files)
        to_keep = set(bucket_files[:n_keep])
        to_delete = set(bucket_files[n_keep:])

        keep.update(to_keep)
        delete.update(to_delete)

    # If the target asked for more files in some buckets than we had,
    # we simply keep everything in those buckets (can't synthesize new files).

    # Apply deletions / moves
    if args.apply:
        if args.trash_dir:
            trash = args.trash_dir
            trash.mkdir(parents=True, exist_ok=True)

        for p in sorted(delete):
            if args.trash_dir:
                # preserve relative path
                rel = p.relative_to(folder)
                dest = trash / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dest))
                action = f"MOVED -> {dest}"
            else:
                os.remove(p)
                action = "DELETED"
            decisions.append((p, file_counts[p], action))

        for p in sorted(keep):
            decisions.append((p, file_counts[p], "KEPT"))

    else:
        # Dry run
        for p in sorted(delete):
            decisions.append((p, file_counts[p], "WOULD_DELETE"))
        for p in sorted(keep):
            decisions.append((p, file_counts[p], "KEEP"))

    # Summary
    kept_hist = Counter(file_counts[p] for p in keep)
    del_hist = Counter(file_counts[p] for p in delete)

    print("\n=== Summary ===")
    print(f"Total files: {total_files}")
    print(f"Observed max tables/file: {observed_max}")
    print(f"Target mode: {mode}, max_val: {max_val}")
    print(f"Dry run: {not args.apply}")
    if args.trash_dir:
        print(f"Trash dir: {args.trash_dir}")

    def fmt_hist(h):
        return ", ".join(f"{k}:{h[k]}" for k in range(0, max_val + 1))

    print("\nObserved histogram: " + fmt_hist(observed_hist))
    print("Target histogram:   " + fmt_hist(Counter({k: target_counts.get(k, 0) for k in range(max_val + 1)})))
    print("Kept histogram:     " + fmt_hist(kept_hist))
    print("Deleted histogram:  " + fmt_hist(del_hist))

    # Optional CSV report
    if args.report:
        import csv

        with args.report.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "tables_in_file", "action"])
            for p, cnt, act in decisions:
                w.writerow([str(p), cnt, act])
        print(f"\nReport written to: {args.report}")


if __name__ == "__main__":
    main()
