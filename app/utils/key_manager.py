# =============================================================================
# File: key_manager.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import base64
import os
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
from tinydb import Query, TinyDB

logger = get_logger("key_manager")


class Client:
    def __init__(self, client_id: str, client_secret: str, client_type: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.client_type = client_type


class KeyManager:
    """Manages client credentials using TinyDB with encryption."""

    def __init__(self, db_path: str = None):
        self.db = None
        self.clients = {}
        self._token_cache: Set[str] = set()  # Cache for valid tokens
        self._admin_cache: Set[str] = set()  # Cache for admin client IDs

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

            # Initialize database with error handling
            try:
                self.db = TinyDB(self.db_path)
                logger.info(f"Using clients database: {self.db_path}")
                self.clients_table = self.db.table("clients")
            except (OSError, PermissionError) as db_error:
                if self.db:
                    self.db.close()
                    self.db = None
                logger.error(f"Database access error: {db_error}")
                raise DatabaseConnectionError(f"Cannot access database: {db_error}")
            except Exception as db_error:
                if self.db:
                    self.db.close()
                    self.db = None
                logger.error(f"Database corruption detected: {db_error}")
                # Try to create a new database file
                if os.path.exists(self.db_path):
                    backup_path = f"{self.db_path}.backup"
                    os.rename(self.db_path, backup_path)
                    logger.info(f"Backed up corrupted database to {backup_path}")
                try:
                    self.db = TinyDB(self.db_path)
                    self.clients_table = self.db.table("clients")
                    logger.info(f"Created new database: {self.db_path}")
                except Exception as recovery_error:
                    raise DatabaseCorruptionError(
                        f"Cannot recover database: {recovery_error}"
                    )

            self.encryption_key = self._get_or_create_encryption_key()
            self.fernet = Fernet(self.encryption_key)
            self.load_clients()
        except (DatabaseConnectionError, DatabaseCorruptionError):
            if self.db:
                self.db.close()
            raise
        except Exception as init_error:
            if self.db:
                self.db.close()
            logger.error(f"Critical error initializing KeyManager: {init_error}")
            raise DatabaseConnectionError(
                f"KeyManager initialization failed: {init_error}"
            )

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
        logger.info(f"Generated new encryption key at {key_file}")
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
            # Use cached token parsing
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

            # Insert or update client
            ClientQuery = Query()
            self.clients_table.upsert(
                {
                    "client_id": client_id,
                    "client_secret": encrypted_secret,
                    "type": client_type,
                },
                ClientQuery.client_id == client_id,
            )

            # Update in-memory cache
            self.clients[client_id] = Client(client_id, client_secret, client_type)

            # Update token and admin caches
            token = f"{client_id}|{client_secret}"
            self._token_cache.add(token)
            if client_type == "admin":
                self._admin_cache.add(client_id)

            # Clear LRU cache to ensure fresh parsing
            self._parse_token.cache_clear()

            logger.info(
                "Added/updated client: %s (%s)",
                sanitize_for_log(client_id),
                sanitize_for_log(client_type),
            )
            return True
        except (PermissionError, OSError) as e:
            logger.error(
                "Database access error for client %s: %s",
                sanitize_for_log(client_id),
                str(e),
            )
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
            ClientQuery = Query()
            result = self.clients_table.remove(ClientQuery.client_id == client_id)

            if result:
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
        except (PermissionError, OSError) as e:
            logger.error(
                "Database access error removing client %s: %s",
                sanitize_for_log(client_id),
                str(e),
            )
            return False
        except Exception as e:
            logger.error(
                "Failed to remove client %s: %s", sanitize_for_log(client_id), str(e)
            )
            return False

    def load_clients(self):
        """Load clients from TinyDB."""
        try:
            # Initialize empty clients dict
            self.clients = {}

            # Check if database file exists and is readable
            if not os.path.exists(self.db_path):
                logger.info(
                    f"Database file {self.db_path} does not exist, will be created"
                )
                return

            all_clients = self.clients_table.all()

            if not all_clients:
                logger.info(f"No clients found in database {self.db_path}")
                return

            for client_data in all_clients:
                try:
                    client_id = client_data["client_id"]
                    encrypted_secret = client_data["client_secret"]
                    client_type = client_data.get("type", "api_user")

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
                    logger.error(
                        f"Decryption failed for client {client_data.get('client_id', 'unknown')}"
                    )
                    continue
                except (KeyError, ValueError) as client_error:
                    logger.error(
                        f"Invalid client data for {client_data.get('client_id', 'unknown')}: {client_error}"
                    )
                    continue
                except Exception as client_error:
                    logger.error(
                        f"Failed to load client {client_data.get('client_id', 'unknown')}: {client_error}"
                    )
                    continue

            logger.info(f"Loaded {len(self.clients)} clients from {self.db_path}")
        except (FileNotFoundError, PermissionError) as e:
            logger.error("Database file access error: %s", str(e))
            self.clients = {}
        except (DecryptionError, DatabaseCorruptionError) as e:
            logger.error("Database/encryption error loading clients: %s", str(e))
            self.clients = {}
            self._token_cache.clear()
            self._admin_cache.clear()
            raise
        except Exception as e:
            logger.error("Failed to load clients: %s", str(e))
            self.clients = {}
            self._token_cache.clear()
            self._admin_cache.clear()

    def close(self):
        """Close database connection and clear caches."""
        if self.db:
            self.db.close()
            self.db = None
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
            # Log to console
            logger.warning(f"=== ADMIN CREDENTIALS CREATED ===")
            logger.warning(f"Admin Client ID: {admin_id}")
            logger.warning(f"Admin Secret: {admin_secret}")
            logger.warning(f"Admin Token: {admin_id}|{admin_secret}")
            logger.warning(f"=== SAVE THESE CREDENTIALS ===")

            # Write to admin credentials file
            try:
                import os
                from datetime import datetime

                from app.utils.path_validator import safe_open

                creds_file = "admin_credentials.txt"
                with safe_open(creds_file, os.getcwd(), "w", encoding="utf-8") as f:
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
                        f'curl -H "Authorization: Bearer {admin_id}|{admin_secret}" \\\n'
                    )
                    f.write(f"  http://localhost:19690/api/v1/admin/clients\n")

                logger.warning(
                    f"Admin credentials saved to: {os.path.abspath(creds_file)}"
                )
            except Exception as e:
                logger.error(f"Failed to save admin credentials to file: {e}")
        else:
            logger.error("Failed to create admin user")


# Ensure admin exists on module load
_ensure_admin_exists()
