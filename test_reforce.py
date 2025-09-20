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
sql = get_best_sql_with_voting(DDL, question, num_votes=3, model="gpt-oss:20b", api_hint="local")

print("Generated SQL:")
print(sql)