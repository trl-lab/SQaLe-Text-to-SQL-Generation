import re
import sqlite3
from typing import Any, List, Optional, Sequence, Tuple

def extract_all_blocks(main_content, code_format):
    sql_blocks = []
    start = 0
    
    while True:

        sql_query_start = main_content.find(f"```{code_format}", start)
        if sql_query_start == -1:
            break
        

        sql_query_end = main_content.find("```", sql_query_start + len(f"```{code_format}"))
        if sql_query_end == -1:
            break 

        sql_block = main_content[sql_query_start + len(f"```{code_format}"):sql_query_end].strip()
        sql_blocks.append(sql_block)

        start = sql_query_end + len("```")
    
    return sql_blocks

def extract_code_blocks(text: str, tag: str):
    pattern = rf"```{tag}\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return [match.strip() for match in matches]

def split_sql_statements(sql: str) -> List[str]:
    # Simple splitter that respects semicolons; does not handle CREATE TRIGGER bodies.
    parts: List[str] = []
    acc = []
    in_str = False
    quote = ''
    for ch in sql:
        if ch in ('"', "'"):
            if not in_str:
                in_str = True
                quote = ch
            elif quote == ch:
                in_str = False
        if ch == ';' and not in_str:
            parts.append(''.join(acc))
            acc = []
        else:
            acc.append(ch)
    if acc:
        parts.append(''.join(acc))
    return parts


def quote_ident(name: str) -> str:
    # Minimal quoting for SQLite identifiers
    if re.search(r"[^A-Za-z0-9_]", name) or name.upper() in SQLITE_RESERVED_WORDS:
        return '"' + name.replace('"', '""') + '"'
    return name

SQLITE_RESERVED_WORDS = set(
    "ABORT, ACTION, ADD, AFTER, ALL, ALTER, ANALYZE, AND, AS, ASC, ATTACH, AUTOINCREMENT,"
    " BEFORE, BEGIN, BETWEEN, BY, CASCADE, CASE, CAST, CHECK, COLLATE, COLUMN, COMMIT,"
    " CONFLICT, CONSTRAINT, CREATE, CROSS, CURRENT_DATE, CURRENT_TIME, CURRENT_TIMESTAMP,"
    " DATABASE, DEFAULT, DEFERRABLE, DEFERRED, DELETE, DESC, DETACH, DISTINCT, DROP, EACH,"
    " ELSE, END, ESCAPE, EXCEPT, EXCLUSIVE, EXISTS, EXPLAIN, FAIL, FOR, FOREIGN, FROM, FULL,"
    " GLOB, GROUP, HAVING, IF, IGNORE, IMMEDIATE, IN, INDEX, INDEXED, INITIALLY, INNER,"
    " INSERT, INSTEAD, INTERSECT, INTO, IS, ISNULL, JOIN, KEY, LEFT, LIKE, LIMIT, MATCH,"
    " NATURAL, NO, NOT, NOTNULL, NULL, OF, OFFSET, ON, OR, ORDER, OUTER, PLAN, PRAGMA,"
    " PRIMARY, QUERY, RAISE, RECURSIVE, REFERENCES, REGEXP, REINDEX, RELEASE, RENAME,"
    " REPLACE, RESTRICT, RIGHT, ROLLBACK, ROW, ROWS, SAVEPOINT, SELECT, SET, TABLE, TEMP,"
    " TEMPORARY, THEN, TO, TRANSACTION, TRIGGER, UNION, UNIQUE, UPDATE, USING, VACUUM,"
    " VALUES, VIEW, VIRTUAL, WHEN, WHERE, WITH, WITHOUT".replace(',', '').split()
)

def best_table_match(tables: Sequence[str], table_cols: dict[str, list[str]], toks: Sequence[str]) -> Optional[str]:
    if not tables:
        return None
    scores = []
    tokset = set(toks)
    for t in tables:
        s = 0
        tl = t.lower()
        s += sum(1 for tok in tokset if tok in tl)
        s += sum(1 for c in table_cols.get(t, []) for tok in tokset if tok in c.lower())
        scores.append((s, t))
    scores.sort(reverse=True)
    return scores[0][1] if scores and scores[0][0] > 0 else tables[0]