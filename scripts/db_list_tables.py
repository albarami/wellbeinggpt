"""
List tables in the configured Postgres database.

Usage:
  python scripts/db_list_tables.py
"""

from __future__ import annotations

import os

import psycopg2


def main() -> None:
    dsn = os.getenv("PSYCOPG2_DSN", "postgresql://postgres:1234@localhost:5432/wellbeing_db")
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    cur.execute("SELECT current_database(), current_user, inet_server_addr(), inet_server_port()")
    print("db/user/server:", cur.fetchone())
    cur.execute(
        """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type='BASE TABLE'
          AND table_schema NOT IN ('pg_catalog','information_schema')
        ORDER BY table_schema, table_name
        """
    )
    rows = cur.fetchall()
    print("DSN:", dsn)
    print("TABLES:", len(rows))
    for s, t in rows:
        print(f"- {s}.{t}")
    conn.close()


if __name__ == "__main__":
    main()


