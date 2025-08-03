"""Pytest configuration and fixtures."""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_dict_json():
    """Sample dictionary-of-dictionaries JSON for testing."""
    return {
        "users": {
            "user1": {
                "name": "Alice",
                "email": "alice@example.com",
                "profile": {
                    "age": 30,
                    "city": "New York"
                }
            },
            "user2": {
                "name": "Bob", 
                "email": "bob@example.com",
                "profile": {
                    "age": 25,
                    "city": "San Francisco"
                }
            }
        },
        "settings": {
            "theme": "dark",
            "notifications": True
        }
    }


@pytest.fixture
def sample_list_json():
    """Sample list JSON for testing."""
    return [
        {"id": 1, "name": "Item 1", "value": 100},
        {"id": 2, "name": "Item 2", "value": 200},
        {"id": 3, "name": "Item 3", "value": 300},
    ]


@pytest.fixture
def sample_mixed_json():
    """Sample mixed structure JSON for testing."""
    return {
        "metadata": {
            "version": "1.0",
            "created": "2024-01-01"
        },
        "data": [
            {"type": "A", "values": [1, 2, 3]},
            {"type": "B", "values": [4, 5, 6]},
        ],
        "config": {
            "enabled": True,
            "settings": {
                "timeout": 30,
                "retries": 3
            }
        }
    }


@pytest.fixture
def large_json_data():
    """Generate large JSON data for size testing."""
    data = {}
    for i in range(100):
        data[f"section_{i}"] = {
            f"item_{j}": f"value_{j}_" + "x" * 100
            for j in range(50)
        }
    return data