import json
import csv
import argparse
from collections import Counter

parser = argparse.ArgumentParser(description="Extract questions from a JSONL file to a CSV file.")
parser.add_argument("input_file", help="Path to the input JSONL file")
parser.add_argument("output_file", help="Path to the output CSV file")
args = parser.parse_args()

questions = []

with open(args.input_file, "r", encoding="utf-8") as infile, open(args.output_file, "w", encoding="utf-8", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["question"])  # CSV header

    for line in infile:
        if not line.strip():
            continue  # skip empty lines
        try:
            rec = json.loads(line)
            question = rec.get("question", "")
            questions.append(question)
            writer.writerow([question])
        except json.JSONDecodeError:
            print("Skipping invalid JSON line:", line.strip())

# Count duplicates
counter = Counter(questions)
duplicates = {q: c for q, c in counter.items() if c > 1 and q}

if duplicates:
    print("\nQuestions that appear more than once:")
    for q, c in duplicates.items():
        print(f'"{q}" appears {c} times')
else:
    print("\nNo duplicate questions found.")

print(f"\n✅ Extracted questions saved to {args.output_file}")
