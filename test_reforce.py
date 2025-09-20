from ReFoRCE.agent import REFORCE
from ReFoRCE.best_sql import get_best_sql_with_voting

DDL = """
CREATE TABLE orders(
  id INTEGER PRIMARY KEY,
  customer TEXT,
  total REAL,
  created_at TEXT
);
CREATE TABLE items(
  order_id INTEGER,
  product TEXT,
  qty INTEGER,
  price REAL
);
"""

question = "List the top 5 customers by total spend."

# One-liner
sql = get_best_sql_with_voting(DDL, question, num_votes=3, model="qwen3:30b", api_hint="local")

print("Generated SQL:")
print(sql)

# Or, if you already constructed a REFORCE with your own chat sessions:
# engine = REFORCE(...existing init...)
# sql = engine.get_sql_from_schema(question, DDL)
