from __future__ import annotations

import re
import sqlite3
from typing import Any, Iterable, List, Optional, Sequence, Tuple
from My_ReFoRCE.utils import quote_ident, split_sql_statements

class InMemoryDB:
    def __init__(self, schema: Any):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        cur = self.conn.cursor()
        self._init_schema(schema, cur)
        self.conn.commit()
        cur.close()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def _init_schema(self, schema: Any, cur: sqlite3.Cursor):
        if isinstance(schema, str):
            ddl = schema
            # Remove custom dialect leftovers, keep SQLite-only constructs
            ddl = self._strip_non_sqlite(ddl)
            cur.executescript(ddl)
            # statements = split_sql_statements(ddl)
            # print("Number of DDL statements:", len(statements))
            # for stmt in statements:
            #     s = stmt.strip()
            #     if not s:
            #         continue
            #    self.conn.execute(s)
        else:
            raise TypeError("schema must be a DDL string with CREATE TABLE statements")

    def _strip_non_sqlite(self, ddl: str) -> str:
        # very light normalization: drop backticks/quotes, dialect keywords we often see
        ddl = ddl.replace("`", "\"")
        ddl = re.sub(r"(?i)CREATE\s+SCHEMA\s+\w+;?", "", ddl)
        ddl = re.sub(r"(?i)CREATE\s+DATABASE\s+\w+;?", "", ddl)
        return ddl

    def table_names(self) -> List[str]:
        cur = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1;")
        return [r[0] for r in cur.fetchall()]

    def columns(self, table: str) -> List[Tuple[str, str]]:
        cur = self.conn.execute(f"PRAGMA table_info({quote_ident(table)});")
        return [(r[1], r[2]) for r in cur.fetchall()]

    def sample_select(self, table: str, cols: Sequence[str] | None = None, limit: int = 20) -> str:
        if cols is None:
            cols = [c for c, _ in self.columns(table)]
        col_list = ", ".join(map(quote_ident, cols)) if cols else "*"
        return f"SELECT {col_list} FROM {quote_ident(table)} LIMIT {int(limit)};"

    def try_exec(self, sql: str) -> Tuple[bool, Optional[str]]:
        try:
            # execute all statements in the SQL string; discard results
            for stmt in split_sql_statements(sql):
                s = stmt.strip()
                if not s:
                    continue
                _ = self.conn.execute(s)
            return True, None
        except Exception as e:
            return False, str(e)