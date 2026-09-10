from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MemoryItem:
    role: str
    text: str
    created_at: str


class JarvisMemory:
    def __init__(self, db_path: str = "data/jarvis-memory.sqlite3") -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute(
            """
            create table if not exists memory (
              id integer primary key autoincrement,
              role text not null,
              text text not null,
              metadata text,
              created_at text not null default current_timestamp
            )
            """
        )
        self.db.commit()

    def add(self, role: str, text: str, metadata: dict | None = None) -> None:
        self.db.execute(
            "insert into memory(role,text,metadata) values(?,?,?)",
            (role, text, json.dumps(metadata or {}, ensure_ascii=False)),
        )
        self.db.commit()

    def recent(self, limit: int = 12) -> list[MemoryItem]:
        rows = self.db.execute(
            "select role,text,created_at from memory order by id desc limit ?", (limit,)
        ).fetchall()
        return [MemoryItem(*row) for row in reversed(rows)]
