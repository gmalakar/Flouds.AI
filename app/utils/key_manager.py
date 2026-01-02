# =============================================================================
# File: key_manager.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import base64
import os
import sqlite3
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional, Set

from cryptography.fernet import Fernet

from app.exceptions import (
    DatabaseConnectionError,
    DatabaseCorruptionError,
    DecryptionError,
)
from app.logger import get_logger
from app.utils.log_sanitizer import sanitize_for_log

logger = get_logger("key_manager")


class Client:
    def __init__(self, client_id: str, client_secret: str, client_type: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.client_type = client_type


class KeyManager:
    """Manages client credentials using SQLite with encryption."""

    def __init__(self, db_path: str = None):
        self.db_path = None
        self.clients = {}
        self._token_cache: Set[str] = set()
        self._admin_cache: Set[str] = set()

        try:
            from app.app_init import APP_SETTINGS

            self.db_path = db_path or getattr(
                APP_SETTINGS.security, "clients_db_path", "clients.db"
            )

            # Ensure directory exists
            db_dir = os.path.dirname(os.path.abspath(self.db_path))
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                logger.info(f"Created database directory: {db_dir}")

            # Initialize database schema
            self._init_database()
            logger.info("Using clients database: %s", sanitize_for_log(self.db_path))

            self.encryption_key = self._get_or_create_encryption_key()
            self.fernet = Fernet(self.encryption_key)
            self.load_clients()

        except (DatabaseConnectionError, DatabaseCorruptionError):
            raise
        except Exception as init_error:
            logger.error(f"Critical error initializing KeyManager: {init_error}")
            raise DatabaseConnectionError(
                f"KeyManager initialization failed: {init_error}"
            )

    @contextmanager
    def _get_connection(self):
        """Get database connection with automatic commit/rollback."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row  # Access columns by name
            yield conn
            conn.commit()
        except sqlite3.OperationalError as e:
            if conn:
                conn.rollback()
            logger.error(f"Database access error: {e}")
            raise DatabaseConnectionError(f"Cannot access database: {e}")
        except sqlite3.DatabaseError as e:
            if conn:
                conn.rollback()
            logger.error(f"Database corruption: {e}")
            raise DatabaseCorruptionError(f"Database corruption detected: {e}")
        except Exception as e:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def _init_database(self):
        """Initialize database schema."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Create clients table with indexes
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS clients (
                        client_id TEXT PRIMARY KEY,
                        client_secret TEXT NOT NULL,
                        client_type TEXT NOT NULL DEFAULT 'api_user',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # Create index on client_type for faster admin lookups
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_client_type 
                    ON clients(client_type)
                """
                )

                # Create trigger to update updated_at
                cursor.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS update_client_timestamp 
                    AFTER UPDATE ON clients
                    BEGIN
                        UPDATE clients SET updated_at = CURRENT_TIMESTAMP
                        WHERE client_id = NEW.client_id;
                    END
                """
                )

                logger.info("Database schema initialized successfully")

        except DatabaseCorruptionError as e:
            # Handle "file is not a database" error (e.g., TinyDB file)
            logger.warning(f"Database corruption detected: {e}")
            self._recover_database()
        except sqlite3.DatabaseError as e:
            # Handle "file is not a database" error (e.g., TinyDB file)
            error_msg = str(e).lower()
            if (
                "not a database" in error_msg
                or "malformed" in error_msg
                or "corrupt" in error_msg
            ):
                logger.warning(f"Database file is corrupted or wrong format: {e}")
                self._recover_database()
            else:
                raise DatabaseConnectionError(f"Cannot initialize database: {e}")
        except sqlite3.OperationalError as e:
            # Try to recover from corrupted database
            if "malformed" in str(e).lower() or "corrupt" in str(e).lower():
                logger.warning(f"Database operational error: {e}")
                self._recover_database()
            else:
                raise DatabaseConnectionError(f"Cannot initialize database: {e}")

    def _recover_database(self):
        """Attempt to recover from corrupted database."""
        logger.warning("Attempting database recovery...")

        if os.path.exists(self.db_path):
            backup_path = f"{self.db_path}.backup"
            try:
                # Remove old backup if exists
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(self.db_path, backup_path)
                logger.info(f"Backed up corrupted database to {backup_path}")
            except OSError as e:
                logger.error(f"Failed to backup corrupted database: {e}")
                # Try to remove the corrupted file directly
                try:
                    os.remove(self.db_path)
                    logger.info(f"Removed corrupted database: {self.db_path}")
                except OSError as remove_error:
                    logger.error(f"Failed to remove corrupted database: {remove_error}")

        # Recreate database - call directly without recursion risk
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS clients (
                        client_id TEXT PRIMARY KEY,
                        client_secret TEXT NOT NULL,
                        client_type TEXT NOT NULL DEFAULT 'api_user',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_client_type 
                    ON clients(client_type)
                """
                )

                cursor.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS update_client_timestamp 
                    AFTER UPDATE ON clients
                    BEGIN
                        UPDATE clients SET updated_at = CURRENT_TIMESTAMP
                        WHERE client_id = NEW.client_id;
                    END
                """
                )

                logger.info("Database recreated successfully")
        except Exception as e:
            raise DatabaseCorruptionError(f"Cannot recover database: {e}")

    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key from environment or file."""
        key_env = os.getenv("FLOUDS_ENCRYPTION_KEY")
        if key_env:
            return base64.urlsafe_b64decode(key_env.encode())

        key_dir = os.path.dirname(os.path.abspath(self.db_path))
        key_file = os.path.join(key_dir, ".encryption_key")

        if os.path.exists(key_file):
            from app.utils.path_validator import safe_open

            with safe_open(key_file, key_dir, "rb") as f:
                return f.read()

        # Generate new key
        key = Fernet.generate_key()
        os.makedirs(key_dir, exist_ok=True)

        from app.utils.path_validator import safe_open

        with safe_open(key_file, key_dir, "wb") as f:
            f.write(key)

        logger.info("Generated new encryption key at %s", sanitize_for_log(key_file))
        return key

    @lru_cache(maxsize=1000)
    def _parse_token(self, token: str) -> Optional[tuple[str, str]]:
        """Parse and cache token parsing results."""
        if "|" not in token:
            return None
        try:
            client_id, client_secret = token.split("|", 1)
            return (client_id, client_secret)
        except (ValueError, IndexError):
            return None

    def authenticate_client(self, token: str) -> Optional[Client]:
        """Authenticate client using client_id|client_secret format."""
        try:
            parsed = self._parse_token(token)
            if not parsed:
                return None

            client_id, client_secret = parsed
            client = self.clients.get(client_id)

            if client and client.client_secret == client_secret:
                return client
            return None
        except (ValueError, TypeError) as e:
            logger.error("Invalid token format: %s", str(e))
            return None
        except Exception as e:
            logger.error("Authentication error: %s", str(e))
            return None

    def is_admin(self, client_id: str) -> bool:
        """Check if client is admin using cache."""
        return client_id in self._admin_cache

    def get_all_tokens(self) -> Set[str]:
        """Get all valid tokens in client_id|client_secret format."""
        return self._token_cache.copy()

    def add_client(
        self, client_id: str, client_secret: str, client_type: str = "api_user"
    ) -> bool:
        """Add new client to database."""
        try:
            # Encrypt secret
            encrypted_secret = self.fernet.encrypt(client_secret.encode()).decode()

            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Use INSERT OR REPLACE for upsert behavior
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO clients (client_id, client_secret, client_type)
                    VALUES (?, ?, ?)
                """,
                    (client_id, encrypted_secret, client_type),
                )

            # Update in-memory cache
            self.clients[client_id] = Client(client_id, client_secret, client_type)

            # Update token and admin caches
            token = f"{client_id}|{client_secret}"
            self._token_cache.add(token)
            if client_type == "admin":
                self._admin_cache.add(client_id)

            # Clear LRU cache
            self._parse_token.cache_clear()

            logger.info(
                "Added/updated client: %s (%s)",
                sanitize_for_log(client_id),
                sanitize_for_log(client_type),
            )
            return True

        except (DatabaseConnectionError, DatabaseCorruptionError):
            return False
        except (ValueError, TypeError) as e:
            logger.error(
                "Invalid client data for %s: %s", sanitize_for_log(client_id), str(e)
            )
            return False
        except Exception as e:
            logger.error(
                "Failed to add client %s: %s", sanitize_for_log(client_id), str(e)
            )
            return False

    def remove_client(self, client_id: str) -> bool:
        """Remove client from database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM clients WHERE client_id = ?", (client_id,))
                deleted = cursor.rowcount > 0

            if deleted:
                # Remove from caches
                removed_client = self.clients.pop(client_id, None)
                if removed_client:
                    token = f"{client_id}|{removed_client.client_secret}"
                    self._token_cache.discard(token)
                    self._admin_cache.discard(client_id)

                # Clear LRU cache
                self._parse_token.cache_clear()

                logger.info(f"Removed client: {client_id}")
                return True
            return False

        except (DatabaseConnectionError, DatabaseCorruptionError):
            return False
        except Exception as e:
            logger.error(
                "Failed to remove client %s: %s", sanitize_for_log(client_id), str(e)
            )
            return False

    def load_clients(self):
        """Load clients from SQLite database."""
        try:
            self.clients = {}

            if not os.path.exists(self.db_path):
                logger.info(
                    f"Database file {self.db_path} does not exist, will be created"
                )
                return

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT client_id, client_secret, client_type 
                    FROM clients
                """
                )

                rows = cursor.fetchall()

                if not rows:
                    logger.info(f"No clients found in database {self.db_path}")
                    return

                for row in rows:
                    try:
                        client_id = row["client_id"]
                        encrypted_secret = row["client_secret"]
                        client_type = row["client_type"]

                        # Decrypt secret
                        try:
                            client_secret = self.fernet.decrypt(
                                encrypted_secret.encode()
                            ).decode()
                        except Exception as decrypt_error:
                            logger.error(
                                f"Failed to decrypt client secret for {client_id}: {decrypt_error}"
                            )
                            raise DecryptionError(
                                f"Cannot decrypt client credentials: {decrypt_error}"
                            )

                        client = Client(client_id, client_secret, client_type)
                        self.clients[client_id] = client

                        # Update caches
                        token = f"{client_id}|{client_secret}"
                        self._token_cache.add(token)
                        if client_type == "admin":
                            self._admin_cache.add(client_id)

                    except DecryptionError:
                        logger.error(f"Decryption failed for client {client_id}")
                        continue
                    except (KeyError, ValueError) as client_error:
                        logger.error(
                            f"Invalid client data for {client_id}: {client_error}"
                        )
                        continue
                    except Exception as client_error:
                        logger.error(
                            f"Failed to load client {client_id}: {client_error}"
                        )
                        continue

                logger.info(f"Loaded {len(self.clients)} clients from {self.db_path}")

        except (DatabaseConnectionError, DatabaseCorruptionError) as e:
            logger.error("Database error loading clients: %s", str(e))
            self.clients = {}
            self._token_cache.clear()
            self._admin_cache.clear()
            raise
        except Exception as e:
            logger.error("Failed to load clients: %s", str(e))
            self.clients = {}
            self._token_cache.clear()
            self._admin_cache.clear()

    def get_client_stats(self) -> dict:
        """Get statistics about stored clients."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Total clients
                cursor.execute("SELECT COUNT(*) as total FROM clients")
                total = cursor.fetchone()["total"]

                # Clients by type
                cursor.execute(
                    """
                    SELECT client_type, COUNT(*) as count 
                    FROM clients 
                    GROUP BY client_type
                """
                )
                by_type = {
                    row["client_type"]: row["count"] for row in cursor.fetchall()
                }

                return {
                    "total_clients": total,
                    "by_type": by_type,
                    "database_size_bytes": (
                        os.path.getsize(self.db_path)
                        if os.path.exists(self.db_path)
                        else 0
                    ),
                }
        except Exception as e:
            logger.error(f"Failed to get client stats: {e}")
            return {
                "total_clients": len(self.clients),
                "by_type": {},
                "database_size_bytes": 0,
            }

    def close(self):
        """Clear caches (SQLite connections are auto-closed by context manager)."""
        self._token_cache.clear()
        self._admin_cache.clear()
        self._parse_token.cache_clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Global instance
key_manager = KeyManager()


# Initialize with default admin if no admin exists
def _ensure_admin_exists():
    # Check if any admin user exists
    admin_exists = any(
        client.client_type == "admin" for client in key_manager.clients.values()
    )

    if not admin_exists:
        from secrets import token_urlsafe

        admin_id = "admin"
        admin_secret = token_urlsafe(32)

        if key_manager.add_client(admin_id, admin_secret, "admin"):
            # Save credentials to console file
            try:
                # write credentials next to the clients DB (key_dir)
                key_dir = os.path.dirname(os.path.abspath(key_manager.db_path))
                console_path = os.path.join(key_dir, "admin_console.txt")
                with open(console_path, "w", encoding="utf-8") as console_file:
                    console_file.write("=== ADMIN CREDENTIALS CREATED ===\n")
                    console_file.write(f"Admin Client ID: {admin_id}\n")
                    console_file.write(f"Admin Secret: {admin_secret}\n")
                    console_file.write(f"Admin Token: {admin_id}|{admin_secret}\n")
                    console_file.write("=== SAVE THESE CREDENTIALS ===\n")
                logger.warning(f"Admin credentials saved to {console_path}")
            except Exception as e:
                logger.error(f"Failed to save console output: {e}")

            # Write to admin credentials file
            try:
                import os
                from datetime import datetime

                from app.utils.path_validator import safe_open

                # Save credentials next to the clients DB so they are colocated with the DB and key
                key_dir = os.path.dirname(os.path.abspath(key_manager.db_path))
                creds_file = "admin_credentials.txt"
                with safe_open(creds_file, key_dir, "w", encoding="utf-8") as f:
                    f.write(f"Flouds AI Admin Credentials\n")
                    f.write(f"Generated: {datetime.now().isoformat()}\n")
                    f.write(f"\n")
                    f.write(f"Client ID: {admin_id}\n")
                    f.write(f"Client Secret: {admin_secret}\n")
                    f.write(f"\n")
                    f.write(f"Usage:\n")
                    f.write(f"Authorization: Bearer {admin_id}|{admin_secret}\n")
                    f.write(f"\n")
                    f.write(f"Example:\n")
                    f.write(
                        f'curl -H "Authorization: Bearer {admin_id}|{admin_secret}" \\\n+'
                    )
                    f.write(f"  http://localhost:19690/api/v1/admin/clients\n")

                creds_path = os.path.join(key_dir, creds_file)
                logger.warning(f"Admin credentials saved to: {creds_path}")
            except Exception as e:
                logger.error(f"Failed to save admin credentials to file: {e}")
        else:
            logger.error("Failed to create admin user")


# Ensure admin exists on module load
_ensure_admin_exists()
