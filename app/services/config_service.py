# =============================================================================
# File: config_service.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: config_service.py
# DB-backed tenant-scoped configuration service (from FloudsVector.Py)
# =============================================================================
import json
import os
import re
import sqlite3
import stat
import threading
from pathlib import Path
from typing import IO, Any, Callable, Iterable, List, Optional, Union, cast

from cryptography.fernet import Fernet

from app.app_init import APP_SETTINGS
from app.logger import get_logger
from app.utils import path_validator as _path_validator
from app.utils.log_sanitizer import sanitize_for_log

logger = get_logger("config_service")

# Cached Fernet instance (lowercase to avoid constant-redefinition warnings)
_fernet: Optional[Fernet] = None

# In-memory cache for tenant-scoped list values (cors_origins, trusted_hosts)
# Keyed by (config_key, tenant_code) -> List[str]
_CACHE: dict[tuple[str, str], List[str]] = {}
_CACHE_LOCK = threading.Lock()


# Provide a typed alias for `safe_open` so Pylance can reason about its signature.
# We only need a reasonably-accurate callable type for the places we use it
# (open a file path relative to a base dir), so a simple Callable is sufficient.
# Provide a thin typed wrapper around the real `safe_open` implementation.
# This gives Pylance a precise signature while delegating to the module
# implementation at runtime.
_SafeOpen = Callable[[Union[str, Path], Union[str, Path], str], IO[Any]]
# Create a typed view of the module implementation so the type checker
# knows the callable signature (the runtime function remains the original).
safe_open_impl = cast(_SafeOpen, _path_validator.safe_open)


def safe_open_t(
    file_path: Union[str, Path], base_dir: Union[str, Path], mode: str = "r"
) -> IO[Any]:
    return safe_open_impl(file_path, base_dir, mode)


def _get_cached_list(key: str, tenant_code: str) -> List[str]:
    t = tenant_code or ""
    cache_key = (key, t)
    with _CACHE_LOCK:
        if cache_key in _CACHE:
            # return a copy to avoid caller mutating internal cache
            return list(_CACHE[cache_key])

    # Cache miss: read from DB
    if t == "":
        val = _read_list(key)
    else:
        val = _read_list_with_tenant(key, t)

    with _CACHE_LOCK:
        _CACHE[cache_key] = list(val)
    return list(val)


def _invalidate_cache_for_key(key: str, tenant_code: Optional[str] = None) -> None:
    """Invalidate cached entries for `key`.

    If `tenant_code` is None, invalidate all tenants for the key.
    If `tenant_code` is provided (including empty string), invalidate only that tenant.
    """
    with _CACHE_LOCK:
        if tenant_code is None:
            # remove all entries for this key
            to_delete = [k for k in _CACHE.keys() if k[0] == key]
            for k in to_delete:
                del _CACHE[k]
        else:
            cache_key = (key, tenant_code or "")
            if cache_key in _CACHE:
                del _CACHE[cache_key]


def _get_fernet() -> Optional[Fernet]:
    """Return a Fernet instance, creating/reading a local key file if necessary."""
    global _fernet
    if _fernet:
        return _fernet

    # Prefer explicit env var
    key_env = os.getenv("FLOUDS_ENCRYPTION_KEY")
    if key_env:
        try:
            key_env_bytes = key_env.encode()
            _fernet = Fernet(key_env_bytes)
            return _fernet
        except Exception:
            logger.exception("Invalid FLOUDS_ENCRYPTION_KEY environment value")
            return None

    # Fall back to .encryption_key in same dir as DB
    try:
        db = _get_db_path()
        key_dir = os.path.dirname(os.path.abspath(db))
        key_file = os.path.join(key_dir, ".encryption_key")
        if os.path.exists(key_file):
            with safe_open_t(key_file, key_dir, "rb") as f:
                key_file_bytes: bytes = f.read()
            _fernet = Fernet(key_file_bytes)
            return _fernet
        # generate and persist a new key
        new_key = Fernet.generate_key()
        os.makedirs(key_dir, exist_ok=True)
        with safe_open_t(key_file, key_dir, "wb") as f:
            f.write(new_key)
        # Set restrictive file permissions (owner read/write only: 0600)
        # This prevents unauthorized access to the encryption key
        os.chmod(key_file, stat.S_IRUSR | stat.S_IWUSR)
        logger.info(
            "Generated new encryption key for config at %s with secure permissions (0600)",
            sanitize_for_log(key_file),
        )
        _fernet = Fernet(new_key)
        return _fernet
    except Exception:
        logger.exception("Failed to initialize encryption key for config service")
        return None


def _get_db_path() -> str:
    return APP_SETTINGS.security.clients_db_path


def init_db() -> None:
    """Ensure the config_kv table exists in the configured SQLite DB."""
    db = _get_db_path()
    conn = None
    try:
        conn = sqlite3.connect(db, timeout=5)
        with conn:
            # Create table with composite primary key (key, tenant_code).
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS config_kv (
                    key TEXT NOT NULL,
                    tenant_code TEXT NOT NULL DEFAULT '',
                    value TEXT NOT NULL,
                    encrypted_flag INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(key, tenant_code)
                )
                """
            )
            # Explicit composite index for efficient lookups by (key, tenant_code).
            # Note: PRIMARY KEY creates a unique index implicitly, but an explicit
            # index makes intent clear and ensures availability for older SQLite
            # versions or tooling that expects the index name.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_config_key_tenant ON config_kv(key, tenant_code)"
            )
            # Migrate from legacy `security_kv` table if present
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='security_kv'"
            )
            if cur.fetchone():
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO config_kv(key, tenant_code, value, encrypted_flag) SELECT key, '', value, 0 FROM security_kv"
                    )
                    conn.execute("DROP TABLE IF EXISTS security_kv")
                    logger.info("Migrated legacy security_kv to config_kv")
                except Exception:
                    logger.exception("Failed to migrate security_kv to config_kv")
    except Exception as e:
        logger.exception(f"Failed to initialize config DB at {db}: {e}")
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass
    # Clear in-memory cache when initializing the DB so tests and fresh
    # environments don't reuse cached values from previous runs.
    with _CACHE_LOCK:
        _CACHE.clear()


def _read_kv(key: str) -> Optional[str]:
    db = _get_db_path()
    conn = None
    try:
        conn = sqlite3.connect(db, timeout=5)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT value, encrypted_flag FROM config_kv WHERE key=? AND tenant_code=?",
            (key, ""),
        )
        row = cur.fetchone()
        if not row:
            return None
        val = row["value"]
        enc = bool(row["encrypted_flag"])
        if enc:
            f = _get_fernet()
            if not f:
                logger.error("Encrypted value found but no encryption key available")
                return None
            try:
                return f.decrypt(val.encode()).decode()
            except Exception:
                logger.exception("Failed to decrypt config value for key %s", key)
                return None
        return val
    except Exception as e:
        logger.exception(f"Failed to read key {key} from config DB: {e}")
        return None
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


def _write_kv(key: str, value: str) -> None:
    db = _get_db_path()
    conn = None
    try:
        conn = sqlite3.connect(db, timeout=5)
        with conn:
            conn.execute(
                "INSERT INTO config_kv(key, tenant_code, value, encrypted_flag) VALUES(?, ?, ?, 0) ON CONFLICT(key, tenant_code) DO UPDATE SET value=excluded.value, encrypted_flag=excluded.encrypted_flag",
                (key, "", value),
            )
    except Exception as e:
        logger.exception(f"Failed to write key {key} into config DB: {e}")
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


def _read_list(key: str) -> List[str]:
    raw = _read_kv(key)
    if not raw:
        return []
    try:
        val: Any = json.loads(raw)
        if isinstance(val, list):
            val_list = val
            return [str(x) for x in val_list]
    except Exception as e:
        logger.warning(f"Invalid JSON for key {key}: {e}")
    return []


def _write_list(key: str, items: List[str]) -> None:
    _write_kv(key, json.dumps(items))


# Public helpers for commonly used keys


def get_cors_origins(tenant_code: str = "") -> List[str]:
    return _get_cached_list("cors_origins", tenant_code)


def set_cors_origins(origins: List[str], tenant_code: str = "", encrypted: bool = False) -> None:
    if tenant_code == "":
        if encrypted:
            _write_encrypted_kv("cors_origins", json.dumps(origins))
        else:
            _write_list("cors_origins", origins)
    else:
        if encrypted:
            _write_encrypted_kv_with_tenant("cors_origins", json.dumps(origins), tenant_code)
        else:
            _write_list_with_tenant("cors_origins", origins, tenant_code)
    # refresh cache for this key/tenant
    _invalidate_cache_for_key("cors_origins", tenant_code)


def get_trusted_hosts(tenant_code: str = "") -> List[str]:
    return _get_cached_list("trusted_hosts", tenant_code)


def set_trusted_hosts(hosts: List[str], tenant_code: str = "", encrypted: bool = False) -> None:
    if tenant_code == "":
        if encrypted:
            _write_encrypted_kv("trusted_hosts", json.dumps(hosts))
        else:
            _write_list("trusted_hosts", hosts)
    else:
        if encrypted:
            _write_encrypted_kv_with_tenant("trusted_hosts", json.dumps(hosts), tenant_code)
        else:
            _write_list_with_tenant("trusted_hosts", hosts, tenant_code)
    # refresh cache for this key/tenant
    _invalidate_cache_for_key("trusted_hosts", tenant_code)


def _read_list_with_tenant(key: str, tenant_code: str) -> List[str]:
    raw = _read_kv_with_tenant(key, tenant_code)
    if not raw:
        return []
    try:
        val: Any = json.loads(raw)
        if isinstance(val, list):
            val_list = val
            return [str(x) for x in val_list]
    except Exception as e:
        logger.warning(f"Invalid JSON for key {key} tenant {tenant_code}: {e}")
    return []


def _write_list_with_tenant(key: str, items: List[str], tenant_code: str) -> None:
    _write_kv_with_tenant(key, json.dumps(items), tenant_code)


# Generic helpers
def get_config(key: str, tenant_code: str = "") -> Optional[str]:
    return _read_kv(key) if tenant_code == "" else _read_kv_with_tenant(key, tenant_code)


def get_config_meta(key: str, tenant_code: str = "") -> tuple[Optional[str], bool]:
    """Return (value, encrypted_flag).

    For encrypted values, the value returned will be None to avoid exposing
    ciphertext or plaintext to callers that should not receive decrypted data.
    The caller can inspect the boolean flag to decide how to respond.
    """
    db = _get_db_path()
    conn = None
    try:
        conn = sqlite3.connect(db, timeout=5)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT value, encrypted_flag FROM config_kv WHERE key=? AND tenant_code=?",
            (key, tenant_code if tenant_code else ""),
        )
        row = cur.fetchone()
        if not row:
            return None, False
        val = row["value"]
        enc = bool(row["encrypted_flag"])
        if enc:
            # don't return ciphertext or decrypted value
            return None, True
        return val, False
    except Exception as e:
        logger.exception(f"Failed to read key {key} tenant {tenant_code} from config DB: {e}")
        return None, False
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


def set_config(key: str, value: str, tenant_code: str = "", encrypted: bool = False) -> None:
    if tenant_code == "":
        if encrypted:
            _write_encrypted_kv(key, value)
        else:
            _write_kv(key, value)
    else:
        if encrypted:
            _write_encrypted_kv_with_tenant(key, value, tenant_code)
        else:
            _write_kv_with_tenant(key, value, tenant_code)
    # Invalidate any cache entries for this key/tenant so middleware sees updates
    _invalidate_cache_for_key(key, tenant_code if tenant_code != "" else "")


def delete_config(key: str, tenant_code: str = "") -> None:
    db = _get_db_path()
    conn = None
    try:
        conn = sqlite3.connect(db, timeout=5)
        with conn:
            conn.execute(
                "DELETE FROM config_kv WHERE key=? AND tenant_code=?",
                (key, tenant_code),
            )
    except Exception as e:
        logger.exception(f"Failed to delete key {key} tenant {tenant_code} from config DB: {e}")
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass
    # Invalidate cache so middleware won't use stale values
    _invalidate_cache_for_key(key, tenant_code if tenant_code != "" else "")


def _read_kv_with_tenant(key: str, tenant_code: str) -> Optional[str]:
    db = _get_db_path()
    conn = None
    try:
        conn = sqlite3.connect(db, timeout=5)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT value, encrypted_flag FROM config_kv WHERE key=? AND tenant_code=?",
            (key, tenant_code),
        )
        row = cur.fetchone()
        if not row:
            return None
        val = row["value"]
        enc = bool(row["encrypted_flag"])
        if enc:
            f = _get_fernet()
            if not f:
                logger.error("Encrypted value found but no encryption key available")
                return None
            try:
                return f.decrypt(val.encode()).decode()
            except Exception:
                logger.exception(
                    "Failed to decrypt config value for key %s tenant %s",
                    key,
                    tenant_code,
                )
                return None
        return val
    except sqlite3.OperationalError as e:
        logger.debug(f"Config DB not available for key {key} tenant {tenant_code}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to read key {key} tenant {tenant_code} from config DB: {e}")
        return None
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


def _write_kv_with_tenant(key: str, value: str, tenant_code: str) -> None:
    db = _get_db_path()
    conn = None
    try:
        conn = sqlite3.connect(db, timeout=5)
        with conn:
            # By default we store plaintext. If caller purposely wants encryption, they should
            # pass an already-encrypted payload and set encrypted_flag in separate helper.
            conn.execute(
                "INSERT INTO config_kv(key, tenant_code, value, encrypted_flag) VALUES(?, ?, ?, 0) ON CONFLICT(key, tenant_code) DO UPDATE SET value=excluded.value, encrypted_flag=excluded.encrypted_flag",
                (key, tenant_code, value),
            )
    except sqlite3.OperationalError as e:
        logger.debug(f"Config DB not available for key {key} tenant {tenant_code}: {e}")
    except Exception as e:
        logger.error(f"Failed to write key {key} tenant {tenant_code} into config DB: {e}")
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


def _write_encrypted_kv(key: str, value: str) -> None:
    """Encrypt `value` and store it for the empty/default tenant with encrypted_flag=1."""
    f = _get_fernet()
    if not f:
        raise RuntimeError("No encryption key available to encrypt config value")
    enc = f.encrypt(value.encode()).decode()
    db = _get_db_path()
    conn = None
    try:
        conn = sqlite3.connect(db, timeout=5)
        with conn:
            conn.execute(
                "INSERT INTO config_kv(key, tenant_code, value, encrypted_flag) VALUES(?, ?, ?, 1) ON CONFLICT(key, tenant_code) DO UPDATE SET value=excluded.value, encrypted_flag=excluded.encrypted_flag",
                (key, "", enc),
            )
    except Exception as e:
        logger.exception(f"Failed to write encrypted key {key} into config DB: {e}")
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


def _write_encrypted_kv_with_tenant(key: str, value: str, tenant_code: str) -> None:
    """Encrypt `value` and store it for the specified tenant with encrypted_flag=1."""
    f = _get_fernet()
    if not f:
        raise RuntimeError("No encryption key available to encrypt config value")
    enc = f.encrypt(value.encode()).decode()
    db = _get_db_path()
    conn = None
    try:
        conn = sqlite3.connect(db, timeout=5)
        with conn:
            conn.execute(
                "INSERT INTO config_kv(key, tenant_code, value, encrypted_flag) VALUES(?, ?, ?, 1) ON CONFLICT(key, tenant_code) DO UPDATE SET value=excluded.value, encrypted_flag=excluded.encrypted_flag",
                (key, tenant_code, enc),
            )
    except Exception as e:
        logger.exception(
            f"Failed to write encrypted key {key} tenant {tenant_code} into config DB: {e}"
        )
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


def load_and_apply_settings() -> None:
    """Read settings from DB and apply them to APP_SETTINGS.security if changed."""
    try:
        # Ensure we read fresh values from DB in case cache is stale
        _invalidate_cache_for_key("cors_origins", None)
        _invalidate_cache_for_key("trusted_hosts", None)

        cors = get_cors_origins()
        trusted = get_trusted_hosts()

        changed = False
        if cors and cors != APP_SETTINGS.security.cors_origins:
            APP_SETTINGS.security.cors_origins = cors
            logger.info(f"Applied CORS origins from DB: {cors}")
            changed = True
        if trusted and trusted != APP_SETTINGS.security.trusted_hosts:
            APP_SETTINGS.security.trusted_hosts = trusted
            logger.info(f"Applied trusted hosts from DB: {trusted}")
            changed = True
        if not changed:
            logger.debug("No config DB changes detected")
    except Exception as e:
        logger.exception(f"Failed to load/apply config settings: {e}")


# Optional helper: list keys for a tenant
def list_keys(tenant_code: str = "") -> List[str]:
    db = _get_db_path()
    conn = None
    try:
        conn = sqlite3.connect(db, timeout=5)
        cur = conn.execute(
            "SELECT key FROM config_kv WHERE tenant_code=? ORDER BY key",
            (tenant_code or "",),
        )
        return [row[0] for row in cur.fetchall()]
    except Exception:
        return []
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# Configuration service providing tenant/global config helpers.
# This mirrors the patterns used in FloudsVector.Py: read settings via
# `ConfigLoader`, normalize lists, and provide helpers to match patterns.
# Cache can be reset in tests.


class ConfigService:
    _cache: Optional[dict[str, List[str]]] = None

    @classmethod
    def _load(cls) -> dict[str, List[str]]:
        try:
            # Prefer values persisted in the config DB via module helpers.
            cors = get_cors_origins()
            trusted = get_trusted_hosts()
            return {"cors_origins": list(cors), "trusted_hosts": list(trusted)}
        except Exception as e:
            logger.exception("Failed to load app settings in ConfigService: %s", e)
            return {"cors_origins": [], "trusted_hosts": []}

    @classmethod
    def reset_cache(cls):
        cls._cache = None

    @classmethod
    def _ensure_cache(cls):
        if cls._cache is None:
            cls._cache = cls._load()

    @classmethod
    def get_cors_origins(cls, tenant_code: str = "") -> List[str]:
        # For tenant-scoped values, defer to DB-backed module helper which
        # supports tenant-specific storage. For the default tenant (empty
        # string), use the cached value for performance.
        if tenant_code:
            return get_cors_origins(tenant_code)
        cls._ensure_cache()
        assert cls._cache is not None
        return list(cls._cache.get("cors_origins", []))

    @classmethod
    def get_trusted_hosts(cls, tenant_code: str = "") -> List[str]:
        if tenant_code:
            return get_trusted_hosts(tenant_code)
        cls._ensure_cache()
        assert cls._cache is not None
        return list(cls._cache.get("trusted_hosts", []))

    @staticmethod
    def _match_pattern(value: str, pattern: Optional[str]) -> bool:
        if pattern is None:
            return False
        val = (value or "").strip()
        pat = pattern.strip()
        if pat == "":
            return True
        if pat == "*":
            return True
        if pat.startswith("re:"):
            try:
                return bool(re.fullmatch(pat[3:], val, flags=re.IGNORECASE))
            except re.error:
                logger.warning("Invalid regex in config: %s", pat)
                return False
        if pat.startswith("*.") or pat.startswith("."):
            root = pat.lstrip("*.").lstrip(".").lower()
            v = val.lower()
            return v == root or v.endswith("." + root)
        if "*" in pat or "?" in pat:
            esc = re.escape(pat)
            esc = esc.replace(r"\*", ".*")
            esc = esc.replace(r"\?", ".")
            try:
                return bool(re.fullmatch(esc, val, flags=re.IGNORECASE))
            except re.error:
                return False
        return val.lower() == pat.lower()

    @classmethod
    def is_allowed(cls, value: str, allowed: Iterable[str]) -> bool:
        for p in allowed:
            if cls._match_pattern(value, p):
                return True
        return False

    # Instance methods delegating to module-level helpers for DB-backed operations
    def init_db(self) -> None:
        # Initialize DB and clear both module-level and class-level caches so
        # subsequent reads reflect the freshly-created DB state.
        init_db()
        # Clear class cache
        self.reset_cache()
        try:
            _invalidate_cache_for_key("cors_origins", None)
            _invalidate_cache_for_key("trusted_hosts", None)
        except Exception:
            pass

    def set_cors_origins(
        self, origins: List[str], tenant_code: str = "", encrypted: bool = False
    ) -> None:
        set_cors_origins(origins, tenant_code=tenant_code, encrypted=encrypted)
        # Invalidate class-level cache so subsequent reads reflect DB changes
        try:
            self.reset_cache()
            _invalidate_cache_for_key("cors_origins", tenant_code if tenant_code != "" else None)
        except Exception:
            pass

    def set_trusted_hosts(
        self, hosts: List[str], tenant_code: str = "", encrypted: bool = False
    ) -> None:
        set_trusted_hosts(hosts, tenant_code=tenant_code, encrypted=encrypted)
        try:
            self.reset_cache()
            _invalidate_cache_for_key("trusted_hosts", tenant_code if tenant_code != "" else None)
        except Exception:
            pass

    def get_config(self, key: str, tenant_code: str = "") -> Optional[str]:
        return get_config(key, tenant_code=tenant_code)

    def set_config(
        self, key: str, value: str, tenant_code: str = "", encrypted: bool = False
    ) -> None:
        set_config(key, value, tenant_code=tenant_code, encrypted=encrypted)
        try:
            self.reset_cache()
            _invalidate_cache_for_key(key, tenant_code if tenant_code != "" else None)
        except Exception:
            pass

    def delete_config(self, key: str, tenant_code: str = "") -> None:
        delete_config(key, tenant_code=tenant_code)
        try:
            self.reset_cache()
            _invalidate_cache_for_key(key, tenant_code if tenant_code != "" else None)
        except Exception:
            pass

    def get_config_meta(self, key: str, tenant_code: str = "") -> tuple[Optional[str], bool]:
        return get_config_meta(key, tenant_code=tenant_code)

    def load_and_apply_settings(self) -> None:
        return load_and_apply_settings()


config_service = ConfigService()
