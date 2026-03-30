"""SQLite 持久化存储"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    cache_key: str
    key_embedding: list[float]
    app: str
    activity: str
    action_type: str            # "TAP" / "SWIPE" / ...
    x: int                      # 归一化坐标 (0-999)
    y: int
    end_x: int | None = None    # swipe 终点
    end_y: int | None = None
    region_hash: str = ""       # 坐标周围区域的 pHash
    created_at: float = 0.0
    hit_count: int = 0
    last_hit_at: float = 0.0


class CacheStore:
    """SQLite 缓存存储"""

    def __init__(self, db_path: str = ".cache/action_cache.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS action_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT NOT NULL,
                key_embedding TEXT NOT NULL,
                app TEXT NOT NULL,
                activity TEXT NOT NULL,
                action_type TEXT NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                end_x INTEGER,
                end_y INTEGER,
                region_hash TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0,
                last_hit_at REAL NOT NULL DEFAULT 0.0
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_app
            ON action_cache (app)
        """)
        self._conn.commit()

    def query(self, app: str, activity: str = "") -> list[CacheEntry]:
        """按 app 查询候选条目（activity 仅记录，不参与过滤）"""
        cursor = self._conn.execute(
            "SELECT * FROM action_cache WHERE app = ?",
            (app,),
        )
        return [self._row_to_entry(row) for row in cursor.fetchall()]

    def upsert(self, entry: CacheEntry):
        """插入或更新缓存条目（按 cache_key + app 去重）"""
        existing = self._conn.execute(
            "SELECT id FROM action_cache WHERE cache_key = ? AND app = ?",
            (entry.cache_key, entry.app),
        ).fetchone()

        if existing:
            self._conn.execute("""
                UPDATE action_cache SET
                    key_embedding = ?, action_type = ?,
                    x = ?, y = ?, end_x = ?, end_y = ?,
                    region_hash = ?, created_at = ?,
                    hit_count = ?, last_hit_at = ?
                WHERE id = ?
            """, (
                json.dumps(entry.key_embedding), entry.action_type,
                entry.x, entry.y, entry.end_x, entry.end_y,
                entry.region_hash, entry.created_at,
                entry.hit_count, entry.last_hit_at,
                existing["id"],
            ))
        else:
            self._conn.execute("""
                INSERT INTO action_cache
                (cache_key, key_embedding, app, activity, action_type,
                 x, y, end_x, end_y, region_hash, created_at, hit_count, last_hit_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.cache_key, json.dumps(entry.key_embedding),
                entry.app, entry.activity, entry.action_type,
                entry.x, entry.y, entry.end_x, entry.end_y,
                entry.region_hash, entry.created_at,
                entry.hit_count, entry.last_hit_at,
            ))
        self._conn.commit()

    def update_hit(self, cache_key: str, app: str, activity: str = ""):
        """更新命中计数"""
        now = time.time()
        self._conn.execute("""
            UPDATE action_cache
            SET hit_count = hit_count + 1, last_hit_at = ?
            WHERE cache_key = ? AND app = ?
        """, (now, cache_key, app))
        self._conn.commit()

    def cleanup_expired(self, ttl_days: int = 30):
        """清理过期条目"""
        cutoff = time.time() - ttl_days * 86400
        self._conn.execute(
            "DELETE FROM action_cache WHERE created_at < ?", (cutoff,)
        )
        self._conn.commit()

    def count(self) -> int:
        """返回条目总数"""
        cursor = self._conn.execute("SELECT COUNT(*) FROM action_cache")
        return cursor.fetchone()[0]

    def close(self):
        self._conn.close()

    def _row_to_entry(self, row: sqlite3.Row) -> CacheEntry:
        return CacheEntry(
            cache_key=row["cache_key"],
            key_embedding=json.loads(row["key_embedding"]),
            app=row["app"],
            activity=row["activity"],
            action_type=row["action_type"],
            x=row["x"],
            y=row["y"],
            end_x=row["end_x"],
            end_y=row["end_y"],
            region_hash=row["region_hash"],
            created_at=row["created_at"],
            hit_count=row["hit_count"],
            last_hit_at=row["last_hit_at"],
        )
