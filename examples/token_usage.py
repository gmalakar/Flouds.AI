#!/usr/bin/env python3
# =============================================================================
# File: token_usage.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import os

import requests

from app.utils.token_generator import generate_api_token


def demo_token_usage():
    """Demonstrate how to generate and use API tokens."""

    # 1. Generate a token
    token = generate_api_token(32)
    print(f"Generated token: {token}")

    # 2. Set up environment (you would do this in your .env file)
    os.environ["FLOUDS_SECURITY_ENABLED"] = "true"
    os.environ["FLOUDS_API_KEY"] = token

    # 3. Make authenticated API calls
    base_url = "http://localhost:19690/api/v1"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Example: Summarization request
    summarize_data = {
        "model": "t5-small",
        "input": "This is a long text that needs to be summarized.",
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            f"{base_url}/summarizer/summarize", json=summarize_data, headers=headers
        )
        print(f"Summarization response: {response.status_code}")
        if response.status_code == 200:
            print(f"Result: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("Server not running. Start the server first.")

    # Example: Embedding request
    embed_data = {
        "model": "all-MiniLM-L6-v2",
        "input": "Text to embed",
        "projected_dimension": 128,
    }

    try:
        response = requests.post(
            f"{base_url}/embedder/embed", json=embed_data, headers=headers
        )
        print(f"Embedding response: {response.status_code}")
        if response.status_code == 200:
            print(f"Result: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("Server not running. Start the server first.")


if __name__ == "__main__":
    demo_token_usage()
