import os
import shutil
import argparse
from math import ceil

def split_files(input_dir, output_dir, split_number):
    # Ensure input directory exists
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Create output directory if missing
    os.makedirs(output_dir, exist_ok=True)

    # Collect all files (ignore directories)
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    total_files = len(files)

    if total_files == 0:
        print("No files found in the input directory.")
        return

    # Calculate how many files per folder
    files_per_folder = ceil(total_files / split_number)

    print(f"Distributing {total_files} files into {split_number} folders (~{files_per_folder} per folder).")

    # Split and copy files
    for i in range(split_number):
        subfolder_name = str(i + 1)
        subfolder_path = os.path.join(output_dir, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)

        start_idx = i * files_per_folder
        end_idx = start_idx + files_per_folder
        subset = files[start_idx:end_idx]

        for filename in subset:
            src = os.path.join(input_dir, filename)
            dst = os.path.join(subfolder_path, filename)
            shutil.copy2(src, dst)

        print(f"Folder {subfolder_name}: {len(subset)} files")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split files into numbered subfolders.")
    parser.add_argument("--input", "-i", required=True, help="Path to input directory containing files.")
    parser.add_argument("--output", "-o", required=True, help="Path to output directory where subfolders will be created.")
    parser.add_argument("--split", "-s", type=int, required=True, help="Number of subfolders to create.")

    args = parser.parse_args()

    split_files(args.input, args.output, args.split)
