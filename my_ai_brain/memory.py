"""Small persistent SQLite text memory; intentionally not a vector database."""

import os
import sqlite3
from pathlib import Path


class SQLiteMemory:
    def __init__(self, database: str | Path):
        self.database = str(database)
        self._uri = False
        self._anchor: sqlite3.Connection | None = None
        if self.database == ":memory:":
            self.database = f"file:my_ai_brain_{id(self)}?mode=memory&cache=shared"
            self._uri = True
            self._anchor = sqlite3.connect(self.database, timeout=5.0, uri=True)
        else:
            path = Path(self.database)
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS memories ("
                "id INTEGER PRIMARY KEY, content TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=5.0, uri=self._uri)
        connection.row_factory = sqlite3.Row
        return connection

    def add(self, content: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute("INSERT INTO memories(content) VALUES (?)", (content,))
            return int(cursor.lastrowid)

    def search(self, query: str = "", limit: int = 20) -> list[dict[str, int | str]]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        with self._connect() as connection:
            if query:
                rows = connection.execute(
                    "SELECT id, content, created_at FROM memories "
                    "WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
                    (f"%{query}%", limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT id, content, created_at FROM memories ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]