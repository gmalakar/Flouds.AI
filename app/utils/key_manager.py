# =============================================================================
# File: key_manager.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import base64
import os
from typing import Dict, List, Optional

from cryptography.fernet import Fernet

from app.logger import get_logger
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
            except Exception as db_error:
                logger.error(f"Failed to initialize database: {db_error}")
                # Try to create a new database file
                if os.path.exists(self.db_path):
                    backup_path = f"{self.db_path}.backup"
                    os.rename(self.db_path, backup_path)
                    logger.info(f"Backed up corrupted database to {backup_path}")
                self.db = TinyDB(self.db_path)
                self.clients_table = self.db.table("clients")
                logger.info(f"Created new database: {self.db_path}")

            self.encryption_key = self._get_or_create_encryption_key()
            self.fernet = Fernet(self.encryption_key)
            self.clients: Dict[str, Client] = {}
            self.load_clients()
        except Exception as init_error:
            logger.error(f"Critical error initializing KeyManager: {init_error}")
            # Initialize with minimal working state
            self.clients = {}
            raise

    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key from environment or file."""
        key_env = os.getenv("FLOUDS_ENCRYPTION_KEY")
        if key_env:
            return base64.urlsafe_b64decode(key_env.encode())

        key_dir = os.path.dirname(os.path.abspath(self.db_path))
        key_file = os.path.join(key_dir, ".encryption_key")
        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                return f.read()

        # Generate new key
        key = Fernet.generate_key()
        os.makedirs(key_dir, exist_ok=True)
        with open(key_file, "wb") as f:
            f.write(key)
        logger.info(f"Generated new encryption key at {key_file}")
        return key

    def authenticate_client(self, token: str) -> Optional[Client]:
        """Authenticate client using client_id|client_secret format."""
        try:
            if "|" not in token:
                return None

            client_id, client_secret = token.split("|", 1)
            client = self.clients.get(client_id)

            if client and client.client_secret == client_secret:
                return client
            return None
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None

    def is_admin(self, client_id: str) -> bool:
        """Check if client is admin."""
        client = self.clients.get(client_id)
        return client and client.client_type == "admin"

    def get_all_tokens(self) -> List[str]:
        """Get all valid tokens in client_id|client_secret format."""
        return [f"{c.client_id}|{c.client_secret}" for c in self.clients.values()]

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

            logger.info(f"Added/updated client: {client_id} ({client_type})")
            return True
        except Exception as e:
            logger.error(f"Failed to add client {client_id}: {e}")
            return False

    def remove_client(self, client_id: str) -> bool:
        """Remove client from database."""
        try:
            ClientQuery = Query()
            result = self.clients_table.remove(ClientQuery.client_id == client_id)

            if result:
                self.clients.pop(client_id, None)
                logger.info(f"Removed client: {client_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove client {client_id}: {e}")
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
                    client_secret = self.fernet.decrypt(
                        encrypted_secret.encode()
                    ).decode()

                    self.clients[client_id] = Client(
                        client_id, client_secret, client_type
                    )
                except Exception as client_error:
                    logger.error(
                        f"Failed to load client {client_data.get('client_id', 'unknown')}: {client_error}"
                    )
                    continue

            logger.info(f"Loaded {len(self.clients)} clients from {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to load clients: {e}")
            # Initialize empty clients dict on error
            self.clients = {}


# Global instance
key_manager = KeyManager()


# Initialize with default admin if no admin exists
def _ensure_admin_exists():
    # Check if any admin user exists
    admin_exists = any(
        client.client_type == "admin" for client in key_manager.clients.values()
    )

    if not admin_exists:
        import secrets

        admin_id = "admin"
        admin_secret = secrets.token_urlsafe(32)

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

                creds_file = "admin_credentials.txt"
                with open(creds_file, "w", encoding="utf-8") as f:
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
