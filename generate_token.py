#!/usr/bin/env python3
# =============================================================================
# File: generate_token.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import argparse
import os
import secrets
import sys

# Add app to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from app.modules.key_manager import key_manager


def main():
    parser = argparse.ArgumentParser(description="Manage client tokens for Flouds AI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add client command
    add_parser = subparsers.add_parser("add", help="Add new client")
    add_parser.add_argument("client_id", help="Client ID")
    add_parser.add_argument(
        "--type",
        "-t",
        default="api_user",
        choices=["admin", "api_user"],
        help="Client type",
    )
    add_parser.add_argument(
        "--secret", "-s", help="Custom client secret (auto-generated if not provided)"
    )

    # List clients command
    list_parser = subparsers.add_parser("list", help="List all clients")

    # Remove client command
    remove_parser = subparsers.add_parser("remove", help="Remove client")
    remove_parser.add_argument("client_id", help="Client ID to remove")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "add":
        client_secret = args.secret or secrets.token_urlsafe(32)

        if key_manager.add_client(args.client_id, client_secret, args.type):
            print(f"✅ Client added successfully!")
            print(f"Client ID: {args.client_id}")
            print(f"Client Type: {args.type}")
            print(f"Client Secret: {client_secret}")
            print(f"\n🔑 Token: {args.client_id}|{client_secret}")
            print(f"\n📋 Usage:")
            print(
                f'curl -H "Authorization: Bearer {args.client_id}|{client_secret}" \\'
            )
            print(f"  http://localhost:19690/api/v1/summarize")
        else:
            print(f"❌ Failed to add client: {args.client_id}")

    elif args.command == "list":
        clients = key_manager.clients
        if clients:
            print(f"📋 Registered Clients ({len(clients)}):")
            for client_id, client in clients.items():
                print(f"  • {client_id} ({client.client_type})")
        else:
            print("📭 No clients registered")

    elif args.command == "remove":
        if key_manager.remove_client(args.client_id):
            print(f"✅ Client removed: {args.client_id}")
        else:
            print(f"❌ Client not found: {args.client_id}")


if __name__ == "__main__":
    main()
