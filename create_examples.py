import csv
from datasets import load_dataset
import re

JOIN_RE = re.compile(r"\bjoin\b", re.IGNORECASE)

def count_joins(sql: str) -> int:
    if not sql:
        return 0
    return len(JOIN_RE.findall(sql))

input_file = 'data/examples/spider2.csv'  # Replace with your actual input CSV file path
output_file = 'data/examples.csv'

# Load BIRD dataset questions
dataset = load_dataset("xu3kev/BIRD-SQL-data-train", split="train")
bird_questions = [(item["question"], item["SQL"]) for item in dataset]

# Read spider2 questions
spider2_questions = []
with open(input_file, newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        spider2_questions.append((row['nl_prompt'], row['sql_query']))

# Combine both lists
all_questions = spider2_questions + bird_questions

# Write combined questions and join counts to output CSV
with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['nl_prompt', 'num_joins'])  # Write header
    for question, sql in all_questions:
        num_joins = count_joins(sql)
        writer.writerow([question, num_joins])