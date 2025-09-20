"""
Script to generate the best SQL statement for a given question and schema using the full ReFoRCE mechanism.

Usage:
    python best_sql.py --schema <schema_file> --question "<question>"

Returns the best SQL string for the input question and schema.
"""
import argparse
from ReFoRCE.agent import REFORCE
from ReFoRCE.prompt import Prompts
from ReFoRCE.chat import GPTChat, OllamaChat


def get_best_sql_with_voting(schema: str, question: str, num_votes: int = 3, model: str = "gpt-4o-mini", api_hint: str = "local") -> str:
    """
    Uses the full ReFoRCE mechanism with voting to generate and select the best SQL statement for a given question and schema.
    Automatically creates a temporary SQLite database from the schema string (CREATE TABLE statements).
    """
    import tempfile
    import os
    import sqlite3
    from ReFoRCE.sql import SqlEnv
    prompt_class = Prompts()
    # chat_session = GPTChat(azure=None, model=model)
    chat_session = OllamaChat(model=model)
    # Create a temp directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a temporary SQLite database from the schema string
        db_path = os.path.join(tmpdir, "temp_schema.sqlite")
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(schema)
        finally:
            conn.close()
        agent = REFORCE(db_path, api_hint, tmpdir, prompt_class, sql_env=SqlEnv(), chat_session_pre=chat_session, chat_session=chat_session, log_save_path="vote_log")
        # Generate candidate SQLs
        candidate_sqls = []
        for i in range(num_votes):
            sql = agent.get_sql_from_schema(question, schema)
            candidate_sqls.append(sql)
        # Save candidate SQLs and execute them
        sql_paths = {}
        for idx, sql in enumerate(candidate_sqls):
            sql_file = os.path.join(tmpdir, f"{idx}_result.sql")
            csv_file = os.path.join(tmpdir, f"{idx}_result.csv")
            with open(sql_file, "w") as f:
                f.write(sql)
            # Execute SQL and save result
            agent.sql_env.execute_sql_api(sql, "sqlite", csv_file, sqlite_path=db_path)
            sql_paths[f"{idx}_result.sql"] = f"{idx}_result.csv"
        # Use voting to select the best SQL
        table_info = "Database schema provided below.\n" + schema
        agent.vote_result(tmpdir, args=type("Args", (), {"do_vote": True, "model_vote": True, "final_choose": True, "do_column_exploration": False, "do_self_consistency": False, "do_self_refinement": False, "save_all_results": False, "random_vote_for_tie": False, "revote": False, "overwrite_unfinished": False, "rerun": False, "num_votes": num_votes, "task": None, "generation_model": model, "db_path": db_path, "output_path": tmpdir, "max_iter": 3}), sql_paths=sql_paths, table_info=table_info, task=question)
        # Read the selected SQL
        result_sql_path = os.path.join(tmpdir, "result.sql")
        if os.path.exists(result_sql_path):
            with open(result_sql_path) as f:
                return f.read().strip()
        # Fallback: return the first candidate
        return candidate_sqls[0]