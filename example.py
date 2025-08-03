#!/usr/bin/env python3
"""
Example usage of the JSON Transformer.

This script demonstrates how to use the JSON Transformer to convert
a JSON structure into LLM-readable files.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from src.json_transformer import JSONTransformer


async def main():
    """Main example function."""
    print("JSON Transformer Example")
    print("=" * 50)
    
    # Create sample data
    sample_data = {
        "users": {
            "user_001": {
                "name": "Alice Johnson",
                "email": "alice@example.com",
                "profile": {
                    "age": 30,
                    "city": "New York",
                    "interests": ["reading", "hiking", "photography"]
                },
                "settings": {
                    "theme": "dark",
                    "notifications": True,
                    "privacy": "public"
                }
            },
            "user_002": {
                "name": "Bob Smith",
                "email": "bob@example.com", 
                "profile": {
                    "age": 25,
                    "city": "San Francisco",
                    "interests": ["coding", "gaming", "music"]
                },
                "settings": {
                    "theme": "light",
                    "notifications": False,
                    "privacy": "private"
                }
            }
        },
        "posts": [
            {
                "id": 1,
                "author": "user_001",
                "title": "My First Post",
                "content": "This is my first post on this platform!",
                "tags": ["introduction", "hello"],
                "created_at": "2024-01-01T10:00:00Z"
            },
            {
                "id": 2,
                "author": "user_002", 
                "title": "Learning Python",
                "content": "I've been learning Python and it's amazing!",
                "tags": ["python", "programming", "learning"],
                "created_at": "2024-01-02T14:30:00Z"
            }
        ],
        "config": {
            "version": "1.0.0",
            "features": {
                "user_registration": True,
                "post_comments": True,
                "private_messaging": False
            },
            "limits": {
                "max_posts_per_user": 100,
                "max_post_length": 5000,
                "max_users": 10000
            }
        }
    }
    
    # Convert to JSON string
    json_string = json.dumps(sample_data, indent=2)
    print(f"Original JSON size: {len(json_string)} characters")
    print(f"Sample of original JSON:\n{json_string[:200]}...\n")
    
    # Create transformer with optimized settings
    transformer = JSONTransformer(default_max_size=92160)  # 90KB optimal for LLM processing
    
    # Use temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Output directory: {temp_dir}")
        
        try:
            # Unflatten the JSON
            print("Unflattening JSON...")
            result = await transformer.unflatten(
                json_string=json_string,
                max_size=8192,  # 8KB for demo (smaller to show splitting)
                output_dir=temp_dir
            )
            
            if result.success:
                print(f"✅ Success!")
                print(f"   Files created: {result.file_count}")
                print(f"   Total size: {result.total_size} bytes")
                print(f"   Output directory: {result.output_directory}")
                
                # List created files
                output_path = Path(result.output_directory)
                json_files = sorted(output_path.glob("*.json"))
                
                print(f"\nCreated files:")
                for file_path in json_files:
                    file_size = file_path.stat().st_size
                    print(f"   {file_path.name} ({file_size} bytes)")
                
                # Show content of first shard file
                if json_files:
                    first_file = json_files[0]
                    if not first_file.name.startswith('_'):  # Skip metadata files
                        print(f"\nSample content of {first_file.name}:")
                        with open(first_file, 'r', encoding='utf-8') as f:
                            content = json.load(f)
                            print(json.dumps(content, indent=2)[:500] + "...")
                
                # Show index file if it exists
                index_file = output_path / "_index.json"
                if index_file.exists():
                    print(f"\nIndex file created: {index_file.name}")
                    with open(index_file, 'r', encoding='utf-8') as f:
                        index_content = json.load(f)
                        print(f"   Total shards in index: {len(index_content.get('shards', []))}")
                        print(f"   Root shard: {index_content.get('root_shard', 'N/A')}")
                        print(f"   Data structure: {index_content.get('data_structure', 'N/A')}")
                
            else:
                print(f"❌ Failed to unflatten JSON")
                if result.errors:
                    for error in result.errors:
                        print(f"   Error: {error}")
        
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())