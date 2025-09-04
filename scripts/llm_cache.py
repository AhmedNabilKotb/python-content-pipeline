# llm_cache.py

import json
import os
import random
import sqlite3
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

JsonDict = Dict[str, Any]


class LLMCache:
    """
    A tiny, robust SQLite cache for LLM call responses.

    Keying strategy:
      key = SHA256( f"{version}|{model}|{namespace}\n{stable_json(payload)}" )

    Features:
      • TTL per entry (expiry on read, lazy cleanup) + optional TTL jitter to avoid stampedes.
      • Namespaces (so different subsystems don't collide) + per-model purging.
      • Optional max_rows / max_bytes with LRU-style pruning (by last_access).
      • WAL mode + small concurrency improvements.
      • Optional transparent zlib compression for large responses.
      • JSON helpers, get-or-compute helpers, and simple stats (incl. token/cost rollups from meta).
      • Context-manager support.
      • Export/Import, healthcheck(), optimize() utilities.

    NOTE: stdlib-only and safe to vendor.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = "logs/llm_cache.sqlite3",
        *,
        default_ttl: Optional[float] = None,         # seconds; None => no expiry
        default_namespace: str = "default",
        max_rows: Optional[int] = None,              # e.g., 50_000
        max_bytes: Optional[int] = None,             # e.g., 256 * 1024 * 1024
        version: str = "v1",                         # bump to invalidate all keys
        key_salt: str = "",                          # optional salt to avoid collisions across apps
        compress_threshold: int = 512,               # bytes; responses above this are zlib-compressed
        sqlite_timeout: float = 5.0,                 # connection busy timeout
        wal: bool = True,                            # use Write-Ahead Logging
        ttl_jitter_fraction: float = 0.0,            # e.g., 0.1 => ±10% jitter
    ):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.default_ttl = default_ttl
        self.default_namespace = default_namespace
        self.max_rows = max_rows
        self.max_bytes = max_bytes
        self.version = version
        self.key_salt = key_salt
        self.compress_threshold = max(0, int(compress_threshold))
        self.sqlite_timeout = float(sqlite_timeout)
        self.ttl_jitter_fraction = max(0.0, float(ttl_jitter_fraction))
        self._ops_since_housekeep = 0
        self._lock = threading.Lock()

        # Connect
        self._conn = sqlite3.connect(
            str(self.path),
            timeout=self.sqlite_timeout,
            isolation_level=None,         # autocommit mode
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.execute("PRAGMA synchronous = NORMAL;")
        if wal:
            try:
                self._conn.execute("PRAGMA journal_mode = WAL;")
                self._conn.execute("PRAGMA wal_autocheckpoint = 500;")
            except sqlite3.OperationalError:
                pass
        self._conn.execute("PRAGMA temp_store = MEMORY;")

        # Schema
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                k TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                namespace TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_access REAL NOT NULL,
                ttl REAL,                       -- NULL => no expiry
                response BLOB NOT NULL,         -- may be compressed (zlib)
                compressed INTEGER NOT NULL DEFAULT 0,  -- 0=plain utf-8, 1=zlib-compressed
                meta TEXT                       -- optional JSON metadata about the call
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_namespace ON cache(namespace);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_last_access ON cache(last_access);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_created_at ON cache(created_at);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_ns_model ON cache(namespace, model);")

    # ------------------------------ utils ------------------------------------

    @staticmethod
    def _stable_json(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def _key(self, model: str, payload: JsonDict, namespace: str) -> str:
        import hashlib
        base = f"{self.version}|{self.key_salt}|{model}|{namespace}\n{self._stable_json(payload)}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def _now(self) -> float:
        return time.time()

    def _jittered_ttl(self, ttl: Optional[float]) -> Optional[float]:
        if ttl is None or self.ttl_jitter_fraction <= 0:
            return ttl
        jitter = (random.random() * 2 - 1) * self.ttl_jitter_fraction  # [-f, +f]
        return max(0.0, ttl * (1.0 + jitter))

    def _should_housekeep(self) -> bool:
        self._ops_since_housekeep += 1
        if self._ops_since_housekeep >= 32:
            self._ops_since_housekeep = 0
            return True
        return random.random() < 0.02

    # ------------------------------ core API ---------------------------------

    def get(
        self,
        model: str,
        payload: JsonDict,
        *,
        namespace: Optional[str] = None,
        min_fresh_seconds: float = 0.0,
    ) -> Optional[str]:
        """
        Fetch a cached text response. Returns None on miss/expired.

        min_fresh_seconds > 0 means "consider near-expiry as expired" to force refresh earlier.
        """
        ns = namespace or self.default_namespace
        k = self._key(model, payload, ns)
        now = self._now()

        with self._lock:
            cur = self._conn.execute(
                "SELECT response, compressed, created_at, ttl FROM cache WHERE k=?",
                (k,),
            )
            row = cur.fetchone()
            if not row:
                return None

            response_blob, compressed, created_at, ttl = row
            if ttl is not None:
                if now > (float(created_at) + float(ttl) - float(min_fresh_seconds)):
                    # Expired: delete lazily
                    self._conn.execute("DELETE FROM cache WHERE k=?", (k,))
                    return None

            # Touch last_access
            self._conn.execute("UPDATE cache SET last_access=? WHERE k=?", (now, k))

        if compressed:
            try:
                response_text = zlib.decompress(response_blob).decode("utf-8")
            except Exception:
                # corrupted — drop on read
                with self._lock:
                    self._conn.execute("DELETE FROM cache WHERE k=?", (k,))
                return None
        else:
            try:
                if isinstance(response_blob, (bytes, bytearray)):
                    response_text = response_blob.decode("utf-8")
                else:
                    response_text = str(response_blob)
            except Exception:
                # invalid utf-8 — drop on read
                with self._lock:
                    self._conn.execute("DELETE FROM cache WHERE k=?", (k,))
                return None

        return response_text

    def get_stale(
        self,
        model: str,
        payload: JsonDict,
        *,
        namespace: Optional[str] = None,
    ) -> Tuple[Optional[str], bool]:
        """
        Fetch cached response ignoring TTL. Returns (text_or_None, is_stale: bool).
        Useful for graceful fallback if the fresh read fails.
        """
        ns = namespace or self.default_namespace
        k = self._key(model, payload, ns)

        with self._lock:
            cur = self._conn.execute(
                "SELECT response, compressed, created_at, ttl FROM cache WHERE k=?",
                (k,),
            )
            row = cur.fetchone()
            if not row:
                return None, False

            response_blob, compressed, created_at, ttl = row
            now = self._now()
            is_stale = ttl is not None and now > (float(created_at) + float(ttl))

            # Touch last_access
            self._conn.execute("UPDATE cache SET last_access=? WHERE k=?", (now, k))

        if compressed:
            try:
                response_text = zlib.decompress(response_blob).decode("utf-8")
            except Exception:
                return None, True
        else:
            try:
                response_text = response_blob.decode("utf-8") if isinstance(response_blob, (bytes, bytearray)) else str(response_blob)
            except Exception:
                return None, True

        return response_text, is_stale

    def get_json(
        self,
        model: str,
        payload: JsonDict,
        *,
        namespace: Optional[str] = None,
        min_fresh_seconds: float = 0.0,
    ) -> Optional[JsonDict]:
        txt = self.get(model, payload, namespace=namespace, min_fresh_seconds=min_fresh_seconds)
        if txt is None:
            return None
        try:
            return json.loads(txt)
        except Exception:
            return None

    def put(
        self,
        model: str,
        payload: JsonDict,
        response_text: str,
        *,
        namespace: Optional[str] = None,
        ttl: Optional[float] = None,
        meta: Optional[JsonDict] = None,
    ) -> None:
        ns = namespace or self.default_namespace
        k = self._key(model, payload, ns)
        now = self._now()
        ttl_val = self._jittered_ttl(ttl if ttl is not None else self.default_ttl)

        raw_bytes = response_text.encode("utf-8")
        if self.compress_threshold and len(raw_bytes) >= self.compress_threshold:
            try:
                blob = zlib.compress(raw_bytes, level=6)
                compressed = 1
            except Exception:
                blob = raw_bytes
                compressed = 0
        else:
            blob = raw_bytes
            compressed = 0

        meta_json = self._stable_json(meta) if meta else None

        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO cache (k, model, namespace, created_at, last_access, ttl, response, compressed, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (k, model, ns, now, now, ttl_val, sqlite3.Binary(blob), compressed, meta_json),
            )

        if self._should_housekeep():
            self._housekeep()

    def put_json(
        self,
        model: str,
        payload: JsonDict,
        response_obj: JsonDict,
        *,
        namespace: Optional[str] = None,
        ttl: Optional[float] = None,
        meta: Optional[JsonDict] = None,
    ) -> None:
        self.put(model, payload, self._stable_json(response_obj), namespace=namespace, ttl=ttl, meta=meta)

    def get_or_compute(
        self,
        model: str,
        payload: JsonDict,
        compute_fn: Callable[[], str],
        *,
        namespace: Optional[str] = None,
        ttl: Optional[float] = None,
        refresh: bool = False,
        min_fresh_seconds: float = 0.0,
        meta: Optional[JsonDict] = None,
    ) -> str:
        """
        Convenience helper: fetch from cache or compute and store.
        Set refresh=True to force recompute/overwrite.
        """
        if not refresh:
            cached = self.get(model, payload, namespace=namespace, min_fresh_seconds=min_fresh_seconds)
            if cached is not None:
                return cached

        result = compute_fn()
        self.put(model, payload, result, namespace=namespace, ttl=ttl, meta=meta)
        return result

    def get_or_compute_json(
        self,
        model: str,
        payload: JsonDict,
        compute_fn: Callable[[], JsonDict],
        *,
        namespace: Optional[str] = None,
        ttl: Optional[float] = None,
        refresh: bool = False,
        min_fresh_seconds: float = 0.0,
        meta: Optional[JsonDict] = None,
    ) -> JsonDict:
        cached = None if refresh else self.get_json(model, payload, namespace=namespace, min_fresh_seconds=min_fresh_seconds)
        if cached is not None:
            return cached
        obj = compute_fn()
        self.put_json(model, payload, obj, namespace=namespace, ttl=ttl, meta=meta)
        return obj

    # ---------------------------- maintenance --------------------------------

    def _delete_expired(self) -> int:
        with self._lock:
            now = self._now()
            cur = self._conn.execute(
                "DELETE FROM cache WHERE ttl IS NOT NULL AND (created_at + ttl) < ?",
                (now,),
            )
            return cur.rowcount or 0

    def _enforce_max_rows(self) -> int:
        if not self.max_rows:
            return 0
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM cache")
            count = int(cur.fetchone()[0])
            if count <= self.max_rows:
                return 0
            to_delete = count - int(self.max_rows)
            self._conn.execute(
                """
                DELETE FROM cache
                WHERE k IN (
                    SELECT k FROM cache
                    ORDER BY last_access ASC
                    LIMIT ?
                )
                """,
                (to_delete,),
            )
            return to_delete

    def _enforce_max_bytes(self) -> int:
        if not self.max_bytes:
            return 0
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return 0
        if size <= self.max_bytes:
            return 0

        deleted_total = 0
        target = int(self.max_bytes * 0.9)
        with self._lock:
            while True:
                try:
                    size = self.path.stat().st_size
                except FileNotFoundError:
                    break
                if size <= target:
                    break
                cur = self._conn.execute(
                    "SELECT k FROM cache ORDER BY last_access ASC LIMIT 500"
                )
                keys = [r[0] for r in cur.fetchall()]
                if not keys:
                    break
                self._conn.executemany("DELETE FROM cache WHERE k=?", [(k,) for k in keys])
                deleted_total += len(keys)
        try:
            with self._lock:
                self._conn.execute("VACUUM;")
        except sqlite3.OperationalError:
            pass
        return deleted_total

    def _housekeep(self) -> None:
        try:
            self._delete_expired()
            self._enforce_max_rows()
            self._enforce_max_bytes()
        except Exception:
            # best-effort housekeeping
            pass

    # ----------------------------- admin ops ---------------------------------

    def purge_namespace(self, namespace: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM cache WHERE namespace=?", (namespace,))
            try:
                self._conn.execute("VACUUM;")
            except sqlite3.OperationalError:
                pass
            return cur.rowcount or 0

    def purge_model(self, model: str, namespace: Optional[str] = None) -> int:
        if namespace:
            sql = "DELETE FROM cache WHERE model=? AND namespace=?"
            params = (model, namespace)
        else:
            sql = "DELETE FROM cache WHERE model=?"
            params = (model,)
        with self._lock:
            cur = self._conn.execute(sql, params)
            try:
                self._conn.execute("VACUUM;")
            except sqlite3.OperationalError:
                pass
            return cur.rowcount or 0

    def set_limits(self, *, max_rows: Optional[int] = None, max_bytes: Optional[int] = None) -> None:
        if max_rows is not None:
            self.max_rows = int(max_rows)
        if max_bytes is not None:
            self.max_bytes = int(max_bytes)
        self._housekeep()

    def clear_all(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cache")
            try:
                self._conn.execute("VACUUM;")
            except sqlite3.OperationalError:
                pass

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*), IFNULL(SUM(LENGTH(response)),0) FROM cache")
            count, bytes_sum = cur.fetchone()

            itoks = otoks = 0
            usd = 0.0
            rows = self._conn.execute("SELECT meta FROM cache WHERE meta IS NOT NULL").fetchall()
            for (mjson,) in rows:
                try:
                    m = json.loads(mjson)
                    itoks += int(m.get("input_tokens", 0) or 0)
                    otoks += int(m.get("output_tokens", 0) or 0)
                    usd += float(m.get("usd_cost", 0.0) or 0.0)
                except Exception:
                    continue

            try:
                size_on_disk = self.path.stat().st_size
            except FileNotFoundError:
                size_on_disk = 0

            return {
                "rows": int(count),
                "bytes_in_rows": int(bytes_sum),
                "db_size": int(size_on_disk),
                "path": str(self.path),
                "max_rows": self.max_rows,
                "max_bytes": self.max_bytes,
                "default_ttl": self.default_ttl,
                "input_tokens": itoks,
                "output_tokens": otoks,
                "usd_cost": round(usd, 6),
            }

    # ----------------------------- import/export -----------------------------

    def export_json(self, out_path: Union[str, Path]) -> int:
        """
        Export cache to a JSON lines file (one row per line).
        Large responses are stored base64 when compressed; text when not.
        Returns number of rows exported.
        """
        import base64

        out = Path(out_path)
        n = 0
        with self._lock, out.open("w", encoding="utf-8") as f:
            cur = self._conn.execute(
                "SELECT k, model, namespace, created_at, last_access, ttl, response, compressed, meta FROM cache"
            )
            for row in cur:
                k, model, namespace, created_at, last_access, ttl, response, compressed, meta = row
                rec: Dict[str, Any] = {
                    "k": k,
                    "model": model,
                    "namespace": namespace,
                    "created_at": float(created_at),
                    "last_access": float(last_access),
                    "ttl": ttl,
                    "meta": meta,
                }
                if int(compressed) == 1:
                    rec["response"] = base64.b64encode(response).decode("ascii")
                    rec["compressed"] = 1
                else:
                    # Try to emit UTF-8 text cleanly; if that fails, fall back to b64 but keep compressed=0
                    try:
                        rec["response"] = response.decode("utf-8")
                        rec["compressed"] = 0
                    except Exception:
                        rec["response"] = base64.b64encode(response).decode("ascii")
                        rec["compressed"] = 0
                        rec["encoding"] = "b64"  # hint for future importers
                f.write(self._stable_json(rec) + "\n")
                n += 1
        return n

    def import_json(self, in_path: Union[str, Path], *, overwrite: bool = False) -> int:
        """
        Import cache from a JSON lines file produced by export_json().
        Accepts historical exports that used compressed=2 for raw base64.
        Returns number of rows imported.
        """
        import base64

        p = Path(in_path)
        if not p.exists():
            return 0
        n = 0
        with self._lock, p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    k = rec["k"]
                    if not overwrite:
                        cur = self._conn.execute("SELECT 1 FROM cache WHERE k=?", (k,))
                        if cur.fetchone():
                            continue

                    compressed = int(rec.get("compressed", 0) or 0)
                    resp_field = rec.get("response", "")
                    encoding = rec.get("encoding", "")

                    # Normalize legacy and current forms:
                    # - compressed=1 => zlib-compressed, base64 encoded
                    # - compressed=0 + (text) => plain utf-8 text
                    # - compressed=0 + encoding=b64 => raw base64 of *plain* text
                    # - legacy: compressed=2 => raw base64 of plain bytes (not zlib)
                    if compressed == 1:
                        blob = base64.b64decode(resp_field)
                        comp_flag = 1
                    elif compressed == 0 and encoding == "b64":
                        # plain text carried as b64
                        raw = base64.b64decode(resp_field)
                        try:
                            text = raw.decode("utf-8")
                        except Exception:
                            text = raw.decode("utf-8", "replace")
                        blob = text.encode("utf-8")
                        comp_flag = 0
                    elif compressed == 2:  # legacy
                        raw = base64.b64decode(resp_field)
                        try:
                            text = raw.decode("utf-8")
                        except Exception:
                            text = raw.decode("utf-8", "replace")
                        blob = text.encode("utf-8")
                        comp_flag = 0
                    else:
                        # compressed==0 and resp_field is direct text
                        blob = (resp_field or "").encode("utf-8")
                        comp_flag = 0

                    self._conn.execute(
                        """
                        INSERT OR REPLACE INTO cache
                        (k, model, namespace, created_at, last_access, ttl, response, compressed, meta)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            k,
                            rec.get("model", ""),
                            rec.get("namespace", self.default_namespace),
                            float(rec.get("created_at", self._now())),
                            float(rec.get("last_access", self._now())),
                            rec.get("ttl", None),
                            sqlite3.Binary(blob),
                            comp_flag,
                            rec.get("meta", None),
                        ),
                    )
                    n += 1
                except Exception:
                    continue
        return n

    # --------------------------- health & tuning ------------------------------

    def healthcheck(self) -> Tuple[bool, str]:
        try:
            with self._lock:
                row = self._conn.execute("PRAGMA integrity_check;").fetchone()
            ok = (row and row[0] == "ok")
            return ok, (row[0] if row and isinstance(row[0], str) else "unknown")
        except Exception as e:
            return False, f"error: {e}"

    def optimize(self) -> None:
        with self._lock:
            try:
                self._conn.execute("ANALYZE;")
            except sqlite3.OperationalError:
                pass
            try:
                self._conn.execute("VACUUM;")
            except sqlite3.OperationalError:
                pass

    # --------------------------- convenience ----------------------------------

    def remaining_ttl(
        self,
        model: str,
        payload: JsonDict,
        *,
        namespace: Optional[str] = None,
    ) -> Optional[float]:
        """
        Returns seconds until expiry; None if not found or entry has no TTL.
        """
        ns = namespace or self.default_namespace
        k = self._key(model, payload, ns)
        with self._lock:
            cur = self._conn.execute("SELECT created_at, ttl FROM cache WHERE k=?", (k,))
            row = cur.fetchone()
            if not row:
                return None
            created_at, ttl = row
            if ttl is None:
                return None
            remaining = (float(created_at) + float(ttl)) - self._now()
            return max(0.0, remaining)

    # --------------------------- context manager ------------------------------

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def __enter__(self) -> "LLMCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
