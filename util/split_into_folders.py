#!/usr/bin/env python3
import os
import shutil
import argparse
from math import ceil

def main():
    parser = argparse.ArgumentParser(
        description="Split files of a given type into numbered subfolders."
    )
    parser.add_argument(
        "filetype",
        help="File extension to group (e.g. '.jpg', '.txt', 'csv').",
    )
    parser.add_argument(
        "num_files",
        type=int,
        help="Number of files per subfolder."
    )
    parser.add_argument(
        "--src",
        default=".",
        help="Source directory (default: current directory)."
    )
    parser.add_argument(
        "--dest",
        default="split_output",
        help="Destination base folder (default: ./split_output)."
    )

    args = parser.parse_args()
    filetype = args.filetype if args.filetype.startswith('.') else '.' + args.filetype
    src_dir = os.path.abspath(args.src)
    dest_dir = os.path.abspath(args.dest)

    # Gather all matching files
    all_files = [f for f in os.listdir(src_dir)
                 if os.path.isfile(os.path.join(src_dir, f)) and f.endswith(filetype)]

    if not all_files:
        print(f"No files with extension '{filetype}' found in {src_dir}")
        return

    os.makedirs(dest_dir, exist_ok=True)
    total = len(all_files)
    chunks = ceil(total / args.num_files)

    print(f"Found {total} '{filetype}' files. Splitting into {chunks} folders...")

    for i in range(chunks):
        start = i * args.num_files
        end = start + args.num_files
        subfolder_name = os.path.join(dest_dir, str(i + 1))
        os.makedirs(subfolder_name, exist_ok=True)

        for f in all_files[start:end]:
            shutil.move(os.path.join(src_dir, f), os.path.join(subfolder_name, f))

        print(f"Moved {len(all_files[start:end])} files to {subfolder_name}")

    print("Done!")

if __name__ == "__main__":
    main()
