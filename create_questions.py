import os
import glob
import time
from typing import List, Tuple, Dict, Any
import sqlite3
import csv

NUM_SCHEMAS = 221_171
AVG_QUESTIONS_PER_SCHEMA = 10

def load_example_questions(path: str) -> List[str]:
    questions = set()
    with open(path, "r", newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("questions", "").strip()
            if q:
                questions.add(q)
    return list(questions)

def validate_sql_executable(schema_sql: str) -> bool:
    try:
        with sqlite3.connect(":memory:") as conn:
            conn.executescript(schema_sql)
        return True
    except Exception:
        return False

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

def execution_feedback(sql_text: str, schema_sql: str) -> Dict[str, Any]:
    try:
        with sqlite3.connect(":memory:") as conn:
            conn.executescript(schema_sql)
            conn.execute("EXPLAIN " + sql_text)
        return {"valid": True, "error": None, "plan": None}
    except Exception as err:
        return {"valid": False, "error": str(err), "hints": None}

def generate_sql_with_reforce(llm, schema_sql: str, question: str, max_iters: int = 3) -> str:
    state = {"schema_sql": schema_sql, "question": question}
    for _ in range(max_iters):
        # Placeholder: Replace with actual LLM call
        draft = llm.generate_sql(state, constraints={"STRICT_ANSWER_FORMAT": True, "SCHEMA_AWARE": True})
        feedback = execution_feedback(draft, schema_sql)
        if feedback["valid"]:
            # Placeholder: enforce format
            return draft.strip().rstrip(";") + ";"
        # Placeholder: update state with feedback
    return ""

def is_executable(sql_text: str, schema_sql: str) -> bool:
    return execution_feedback(sql_text, schema_sql)["valid"]

def consistent_with_question(sql_text: str, question: str) -> bool:
    # Placeholder: implement semantic consistency check
    return True

def passes_safety_rules(sql_text: str) -> bool:
    # Placeholder: implement safety checks
    return True

def validate_pair(schema_sql: str, question: str, sql_text: str) -> bool:
    if not sql_text:
        return False
    if not is_executable(sql_text, schema_sql):
        return False
    if not consistent_with_question(sql_text, question):
        return False
    if not passes_safety_rules(sql_text):
        return False
    return True

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
        key = (row[3], row[4])  # question_text, sql_text
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped

def log_stats(dataset: List[Tuple]):
    total_pairs = len(dataset)
    by_cmd = {}
    for row in dataset:
        cmd = row[5]
        by_cmd[cmd] = by_cmd.get(cmd, 0) + 1
    execution_rate = total_pairs / (NUM_SCHEMAS * AVG_QUESTIONS_PER_SCHEMA)
    print({"total_pairs": total_pairs, "by_cmd": by_cmd, "EXECUTION_RATE": execution_rate})

def build_semisynth_dataset(schema_folder: str, example_questions_file: str, llm) -> List[Tuple]:
    exemplars = load_example_questions(example_questions_file)
    dataset = []
    sql_files = glob.glob(os.path.join(schema_folder, "*.sql"))
    for file in sql_files:
        with open(file, "r") as f:
            schema_sql = f.read()
        if not validate_sql_executable(schema_sql):
            continue
        questions = generate_questions_for_schema(llm, file, exemplars, AVG_QUESTIONS_PER_SCHEMA)
        for q in questions:
            sql_candidate = generate_sql_with_reforce(llm, schema_sql, q)
            if not validate_pair(schema_sql, q, sql_candidate):
                continue
            cmd_type = annotate_sql_type(sql_candidate)
            meta = {
                "schema_file": os.path.basename(file),
                "tokens_estimate": estimate_token_usage(q, sql_candidate),
                "source": "semi-synthetic",
                "generation_model": "Qwen3-30B",
                "pipeline": "ReFoRCE",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            dataset.append((os.path.basename(file), schema_sql, meta, q, sql_candidate, cmd_type))
    dataset = dedup_and_sanity_check(dataset)
    log_stats(dataset)
    return dataset

# Placeholder LLM class for demonstration
class DummyLLM:
    def synthesize_list(self, prompt, count):
        return [f"Dummy question {i+1}" for i in range(count)]
    def generate_sql(self, state, constraints):
        return "SELECT * FROM dummy_table;"

# Example usage:
# dataset = build_semisynth_dataset("/path/to/schemas", "/path/to/examples.txt", DummyLLM())