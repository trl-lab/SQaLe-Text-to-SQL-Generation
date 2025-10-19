#!/usr/bin/env python3
"""Create a CSV listing of files in a directory.

Usage:
  python util/create_file_list.py /path/to/folder -o out.csv --recursive

The CSV will have two columns: path,filename
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


def iter_files(directory: Path, recursive: bool) -> Iterable[Path]:
	"""Yield files under directory. If recursive is False, only top-level files are yielded."""
	if not directory.exists():
		raise FileNotFoundError(f"Directory does not exist: {directory}")
	if recursive:
		for p in directory.rglob("*"):
			if p.is_file():
				yield p
	else:
		for p in directory.iterdir():
			if p.is_file():
				yield p


def write_csv(file_paths: Iterable[Path], out_file: Path) -> None:
	"""Write CSV with columns path,filename where path is the parent directory and filename is the name."""
	out_file.parent.mkdir(parents=True, exist_ok=True)
	with out_file.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["path", "filename"])
		for p in file_paths:
			writer.writerow([str(p.parent), p.name])


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Create CSV of files in a directory")
	p.add_argument("directory", type=Path, help="Directory to list files from")
	p.add_argument("-o", "--output", type=Path, default=Path("file_list.csv"), help="Output CSV file")
	p.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	files = list(iter_files(args.directory, args.recursive))
	write_csv(files, args.output)
	print(f"Wrote {len(files)} files to {args.output}")


if __name__ == "__main__":
	main()
