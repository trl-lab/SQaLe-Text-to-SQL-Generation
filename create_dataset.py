import os
import glob
import time
from typing import List, Tuple, Dict, Any
import sqlite3
import csv
from ollama import Client
from ReFoRCE.best_sql import get_best_sql_with_voting
from ReFoRCE.utils import extract_code_blocks
import random
from tqdm import tqdm
import json
import argparse

NUM_SCHEMAS = 221_171
AVG_QUESTIONS_PER_SCHEMA = 10

def load_example_questions(path: str) -> List[str]:
    questions = set()
    with open(path, "r", newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("nl_prompt", "").strip()
            if q:
                questions.add(q)
    return list(questions)

def build_question_prompt(schema_file: str, exemplars: List[str], guidelines: Dict[str, Any]) -> str:
    # Simple prompt builder; can be extended
    prompt = (
        f"Given the schema in {schema_file}, generate {AVG_QUESTIONS_PER_SCHEMA} diverse natural language questions.\n"
        f"Guidelines: {', '.join(guidelines.values())}\n"
        f"Example questions:\n" + "\n".join(exemplars[:5])
    )
    return prompt

def generate_questions_for_schema(llm, schema_file: str, exemplars: List[str], n: int) -> List[str]:
    guidelines = {
        "1": "ask diverse intents",
        "2": "cover SELECT/INSERT/UPDATE/DELETE",
        "3": "use realistic domain language",
        "4": "avoid copying exemplars verbatim"
    }
    prompt = build_question_prompt(schema_file, exemplars, guidelines)
    # Placeholder: Replace with actual LLM call
    return llm.synthesize_list(prompt, count=n)

def generate_sql_with_reforce(llm, schema_sql: str, question: str, max_iters: int = 3) -> str:
    state = {"schema_sql": schema_sql, "question": question}
    for _ in range(max_iters):
        # Placeholder: Replace with actual LLM call
        draft = llm.generate_sql(question, schema_sql)
       
        # Placeholder: enforce format
        return draft.strip().rstrip(";") + ";"
        # Placeholder: update state with feedback
    return ""

def annotate_sql_type(sql_text: str) -> str:
    t = sql_text.strip().upper()
    if t.startswith("SELECT"):
        return "SELECT"
    if t.startswith("INSERT"):
        return "INSERT"
    if t.startswith("UPDATE"):
        return "UPDATE"
    if t.startswith("DELETE"):
        return "DELETE"
    if t.startswith("CREATE"):
        return "CREATE"
    if t.startswith("DROP"):
        return "DROP"
    if t.startswith("ALTER"):
        return "ALTER"
    return "OTHER"

def estimate_token_usage(q: str, sql: str) -> int:
    # Placeholder: simple token estimate
    return len(q.split()) + len(sql.split())

def dedup_and_sanity_check(dataset: List[Tuple]) -> List[Tuple]:
    seen = set()
    deduped = []
    for row in dataset:
        key = (row[1], row[3], row[4])  # schema, question_text, sql_text
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped

def build_semisynth_dataset(schema_folder: str, example_questions_file: str, llm) -> List[Tuple]:
    print("Loading example questions...")
    exemplars = load_example_questions(example_questions_file)
    print(f"Loaded {len(exemplars)} example questions.")
    dataset = []
    sql_files = glob.glob(os.path.join(schema_folder, "*.sql"))
    print(f"Found {len(sql_files)} schema files.")
    try:
        for file in tqdm(sql_files, desc="Processing schemas"):
            with open(file, "r") as f:
                schema_sql = f.read()
            # Random 10 examples for diversity
            random.shuffle(exemplars)
            current_exemplars = exemplars[:10]

            questions = llm.synthesize_list(schema_sql, current_exemplars, AVG_QUESTIONS_PER_SCHEMA)
            questions = questions[:min(len(questions), AVG_QUESTIONS_PER_SCHEMA)]
            print(f"Generated {len(questions)} questions for schema {os.path.basename(file)}.")
            for q in questions:
                if q != "":
                    sql_candidate = generate_sql_with_reforce(llm, schema_sql, q)
                    cmd_type = annotate_sql_type(sql_candidate)
                    meta = {
                        "schema_file": os.path.basename(file),
                        "tokens_estimate": estimate_token_usage(q, sql_candidate),
                        "source": "semi-synthetic",
                        "generation_model": llm.model_name,
                        "pipeline": "ReFoRCE",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    dataset.append((os.path.basename(file), schema_sql, meta, q, sql_candidate, cmd_type))
    except KeyboardInterrupt:
        print("Interrupted! Finalizing dataset...")
    print(f"Generated {len(dataset)} question-SQL pairs before deduplication.")
    dataset = dedup_and_sanity_check(dataset)
    return dataset

class LLM_Interface:
    def __init__(self, model_name: str = "gpt-oss:20b", host: str = None):
        self.model_name = model_name
        self.client = Client(host=os.environ.get("OLLAMA_HOST", host or "http://localhost:11434"))

    def synthesize_list(self, schema, examples, count):
        prompt = "/no_think Generate 10 diverse questions for the following schema:\n"
        prompt += schema + "\n"
        prompt += "Use these examples as inspiration in terms of style and complexity:\n"
        prompt += "\n".join(examples) + "\n"
        prompt += f"Please provide {count} questions."
        prompt += f"Use the following format for your response and put it into the plaintext code box:\n"
        prompt += "```plaintext\n<First Question>\n<2nd Question>\n...\n<Nth Question>\n```"

        response = self.client.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        raw = response["message"]["content"]
        blocks = extract_code_blocks(raw, "plaintext")
        if blocks:
            return [block.strip() for block in blocks][-1].split("\n")
        print(f"Raw response:\n{raw}\n")
        return []

    def generate_sql(self, question, tables):
        return get_best_sql_with_voting(tables, question, num_votes=2, model=self.model_name, api_hint="local")

if __name__ == "__main__":
    print("Building semi-synthetic dataset...")
    parser = argparse.ArgumentParser(description="Build semi-synthetic text-to-SQL dataset.")
    parser.add_argument("--model_name", type=str, default="qwen2.5-coder:7b", help="LLM model name to use")
    args = parser.parse_args()

    dataset = build_semisynth_dataset(
        "data/statements",
        "data/examples.csv",
        LLM_Interface(model_name=args.model_name)
    )

    print(f"Generated {len(dataset)} question-SQL pairs.")
    out_file = "semi_synthetic_dataset.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for row in dataset:
            prompt = " ".join(row[3].splitlines()).strip()
            sql_statement = " ".join(row[4].splitlines()).strip()
            schema = " ".join(row[1].splitlines()).strip()
            cmd_type = " ".join(row[5].splitlines()).strip()
            item = {
                "prompt": prompt,
                "sql_statement": sql_statement,
                "schema": schema,
                "cmd_type": cmd_type
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")