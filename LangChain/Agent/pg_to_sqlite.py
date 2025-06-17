import sqlite3
import psycopg2
from psycopg2.extras import DictCursor

# PostgreSQL connection info
PG_CONN_INFO = "dbname=chinook user=joseandres host=localhost"

# SQLite output file
SQLITE_FILE = "chinook.db"

def get_tables(pg_conn):
    with pg_conn.cursor() as cur:
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type='BASE TABLE';
        """)
        return [row[0] for row in cur.fetchall()]

def get_table_columns(pg_conn, table):
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(f"SELECT * FROM {table} LIMIT 0")
        return [desc.name for desc in cur.description]

def copy_table(pg_conn, sqlite_conn, table):
    columns = get_table_columns(pg_conn, table)
    col_str = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(columns))

    pg_cur = pg_conn.cursor()
    pg_cur.execute(f"SELECT {col_str} FROM {table}")

    rows = pg_cur.fetchall()

    # Create table in SQLite with generic TEXT columns
    sqlite_conn.execute(f"DROP TABLE IF EXISTS {table}")
    create_sql = f"CREATE TABLE {table} ({', '.join([col + ' TEXT' for col in columns])})"
    sqlite_conn.execute(create_sql)

    sqlite_conn.executemany(
        f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})",
        rows
    )
    sqlite_conn.commit()

def main():
    pg_conn = psycopg2.connect(PG_CONN_INFO)
    sqlite_conn = sqlite3.connect(SQLITE_FILE)

    tables = get_tables(pg_conn)
    print(f"Found tables: {tables}")

    for table in tables:
        print(f"Copying table {table} ...")
        copy_table(pg_conn, sqlite_conn, table)

    pg_conn.close()
    sqlite_conn.close()
    print(f"Export finished. SQLite DB created at {SQLITE_FILE}")

if __name__ == "__main__":
    main()
