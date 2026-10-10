#!/usr/bin/env python3
"""
Test MongoDB Atlas connection.

This script verifies connectivity to MongoDB Atlas using credentials from .env.
"""

import os
import sys
from typing import Optional

try:
    from dotenv import load_dotenv
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except ImportError as e:
    print("[ERROR] Missing required dependencies")
    print(f"Error: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)


def load_config() -> tuple[Optional[str], Optional[str]]:
    """Load MongoDB configuration from environment variables."""
    load_dotenv()

    mongodb_uri = os.getenv("MONGODB_URI")
    mongodb_db = os.getenv("MONGODB_DB")

    return mongodb_uri, mongodb_db


def test_connection() -> None:
    """Test connection to MongoDB Atlas."""
    mongodb_uri, mongodb_db = load_config()

    if not mongodb_uri:
        print("MongoDB connection failed")
        print("Error type: ValueError")
        print("Error: Missing MONGODB_URI in .env file")
        sys.exit(1)

    if not mongodb_db:
        print("MongoDB connection failed")
        print("Error type: ValueError")
        print("Error: Missing MONGODB_DB in .env file")
        sys.exit(1)

    client = None
    try:
        # Connect with short timeout for quick feedback
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        # Actually verify connectivity
        client.admin.command("ping")
        print("[SUCCESS] MongoDB Atlas connection successful")
        print(f"Database: {mongodb_db}")
        print("Connection is ready for development.")
        sys.exit(0)
    except PyMongoError as e:
        print("MongoDB connection failed")
        print(f"Error type: {type(e).__name__}")
        # Print safe error message (never expose URI, password, or secrets)
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print("MongoDB connection failed")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    test_connection()
