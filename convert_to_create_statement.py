from datasets import load_dataset
import sqlite3
from pathlib import Path
from tqdm import tqdm
from tqdm import trange

# Load dataset
ds = load_dataset("trl-lab/schemapile", split="full")


sqlite_valid_types = [
    # Core storage classes
    "NULL", "INTEGER", "REAL", "TEXT", "BLOB",

    # Integer affinity
    "INT", "TINYINT", "SMALLINT", "MEDIUMINT", "BIGINT",
    "UNSIGNED BIG INT", "INT2", "INT8", "BOOLEAN", "BOOL",

    # Text affinity
    "CHARACTER", "CHAR", "NCHAR", "NATIVE CHARACTER", "VARCHAR",
    "NVARCHAR", "VARYING CHARACTER", "CLOB", "TEXT",
    "MEDIUMTEXT", "LONGTEXT", "TINYTEXT",

    # Real / numeric affinity
    "NUMERIC", "DECIMAL", "REAL", "DOUBLE", "DOUBLE PRECISION", "FLOAT",

    # Date/time (stored as TEXT, REAL, or INTEGER)
    "DATE", "DATETIME", "TIME", "TIMESTAMP"
]

sqlite_protected_keywords = [
    "ABSTRACT", "ALIAS", "CHECK", "COLLATE", "COLUMN", "CONSTRAINT",
    "CREATE", "DATABASE", "DEFAULT", "DELETE", "DESC", "DISTINCT",
    "DROP", "EXCLUSIVE", "EXISTS", "INDEX", "INDEXED", "INHERIT",
    "JOIN", "KEY", "LIMIT", "MATCH", "NATURAL", "NOT", "PRIMARY",
    "REFERENCES", "SELECT", "TABLE", "TEMP", "TEMPORARY", "UNIQUE",
    "UPDATE", "USING", "VIEW", "TO", "AS", "COMMIT", "BUILD", "VALUES",
    "TRANSACTION", "ORDER", "LIMIT", "OFFSET", "FETCH", "INSERT", "REPLACE",
    "INTO", "IS", "NULL", "LIKE", "GLOB", "REGEXP", "ESCAPE", "EXPLAIN",
    "FROM", "GROUP", "TRANSACTION",
    ]

# Make them all lower
sqlite_valid_types = [t.lower() for t in sqlite_valid_types]
sqlite_protected_keywords = [k.lower() for k in sqlite_protected_keywords]

def json_to_create_tables(record):
    tables = record["TABLES"]
    sql_statements = []

    for table in tables:
        try:

            table_name = table["TABLE_NAME"]
            cols = table["COLUMNS"]
            pk = table.get("PRIMARY_KEY", [])
            fks = table.get("FOREIGN_KEYS", {"COLUMNS": [], "FOREIGN_TABLE": [], "REFERRED_COLUMNS": [], "ON_DELETE": [], "ON_UPDATE": []})

            col_defs = []

            table_name = f'"{table_name}"' if table_name.lower() in sqlite_protected_keywords else table_name

            for columns in cols:
                name = columns.get("NAME", "col")
                col_type = columns.get("TYPE", "TEXT") or "TEXT"
                nullable = columns.get("NULLABLE", True)
                unique = columns.get("UNIQUE", False)
                default = columns.get("DEFAULT", None)

                name = f'"{name}"' if name.lower() in sqlite_protected_keywords else name

                # if the name starts with an integer, put it in quotes
                if name[0].isdigit():
                    name = f'"{name}"'
                if col_type.lower() not in sqlite_valid_types:
                    col_sql = f"{name} TEXT"
                else:
                    col_sql = f"{name} {col_type}"
                if not nullable:
                    col_sql += " NOT NULL"
                if unique:
                    col_sql += " UNIQUE"
                if default is not None and default != "":
                    if not isinstance(default, int):
                        default = default.replace("'", "")
                        col_sql += f" DEFAULT '{default}'"
                    else:
                        col_sql += f" DEFAULT {default}"
                col_defs.append(col_sql)

            # Primary key
            if pk:
                col_defs.append(f"PRIMARY KEY ({', '.join(pk)})")

            for fk in fks:
                
                cols_fk = fk.get("COLUMNS", [])
                ft = fk.get("FOREIGN_TABLE", "")
                ref = fk.get("REFERRED_COLUMNS", [])
                on_delete = fk.get("ON_DELETE", "")
                on_update = fk.get("ON_UPDATE", "")

                on_delete = "SET NULL" if on_delete == "SetNull" else on_delete
                on_delete = "CASCADE" if on_delete == "Cascade" else on_delete
                on_delete = "RESTRICT" if on_delete == "Restrict" else on_delete
                on_delete = "NO ACTION" if on_delete == "NoAction" else on_delete
                on_delete = "SET DEFAULT" if on_delete == "SetDefault" else on_delete

                on_update = "SET NULL" if on_update == "SetNull" else on_update
                on_update = "CASCADE" if on_update == "Cascade" else on_update
                on_update = "RESTRICT" if on_update == "Restrict" else on_update
                on_update = "NO ACTION" if on_update == "NoAction" else on_update
                on_update = "SET DEFAULT" if on_update == "SetDefault" else on_update

                if cols_fk and ft and ref:
                    fk_cols = ", ".join(cols_fk)
                    ref_cols = ", ".join(ref)
                    ft = ft.replace(".", "_")
                    ft = f'"{ft}"' if ft.lower() in sqlite_protected_keywords else ft
                    fk_sql = f"FOREIGN KEY ({fk_cols}) REFERENCES {ft} ({ref_cols})"
                    if on_delete:
                        fk_sql += f" ON DELETE {on_delete}"
                    if on_update:
                        fk_sql += f" ON UPDATE {on_update}"
                    col_defs.append(fk_sql)

            # Build full CREATE TABLE
            table_name = table_name.replace(".", "_")
            table_name = table_name.replace("-", "_")
            table_sql = f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(col_defs) + "\n);"
            sql_statements.append(table_sql)
        except Exception as e:
            print(f"Error generating SQL for table {table_name}: {e}")
            import traceback; print(traceback.format_exc())
            continue

    return sql_statements


# Example: generate SQL for first dataset entry
all_statements = []
for i in trange(len(ds), desc="Generating CREATE statements"):
    sqls = json_to_create_tables(ds[i])
    all_statements.append(sqls)


for i, sqls in enumerate(tqdm(all_statements, desc="Processing statements")):

    try:
        # For convenience, also save the SQL we executed to a .sql file
        sql_path = Path(f"data/statements/schemapile_{i}.sql")
        sql_path.write_text("\n\n".join(sqls))

        # Create the SQLite DB and execute the statements
        db_path = Path(f"data/datasets/schemapile_{i}.sqlite")
        if db_path.exists():
            db_path.unlink()

        con = sqlite3.connect(db_path)
        try:
            con.execute("PRAGMA foreign_keys = ON;")
            cur = con.cursor()
            for s in sqls:
                cur.execute(s)
            con.commit()
        finally:
            con.close()
    except Exception as e:
        # Delete files again
        if sql_path.exists():
            sql_path.unlink(missing_ok=True)
        if db_path.exists():
            db_path.unlink(missing_ok=True)

        print(f"Error processing statements for dataset {i}: {e}")