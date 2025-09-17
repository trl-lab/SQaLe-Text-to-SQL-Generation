import csv
from datasets import load_dataset

input_file = 'data/examples/spider2.csv'  # Replace with your actual input CSV file path
output_file = 'data/examples.csv'

# Load BIRD dataset questions
dataset = load_dataset("xu3kev/BIRD-SQL-data-train", split="train")
bird_questions = [item["question"] for item in dataset]

# Read spider2 questions
spider2_questions = []
with open(input_file, newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        spider2_questions.append(row['nl_prompt'])

# Combine both lists
all_questions = spider2_questions + bird_questions

# Write combined questions to output CSV
with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['nl_prompt'])  # Write header
    for question in all_questions:
        writer.writerow([question])
