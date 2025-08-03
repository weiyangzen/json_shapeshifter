#!/usr/bin/env python3
"""
Advanced example showcasing state-of-the-art JSON Transformer features.

This script demonstrates the advanced capabilities including:
- 90KB optimal chunk sizing
- Performance profiling and optimization
- Streaming processing for large datasets
- Compression optimizations
- Parallel processing
- Memory-efficient algorithms
"""

import asyncio
import json
import tempfile
import logging
from pathlib import Path
from src.json_transformer import JSONTransformer


def create_large_dataset():
    """Create a large, complex dataset for demonstration."""
    return {
        "metadata": {
            "version": "2.0.0",
            "created": "2024-01-01T00:00:00Z",
            "dataset_type": "comprehensive_demo",
            "total_records": 10000,
            "schema_version": "1.2.3"
        },
        "users": {
            f"user_{i:06d}": {
                "id": i,
                "username": f"user_{i}",
                "email": f"user_{i}@example.com",
                "profile": {
                    "first_name": f"FirstName{i}",
                    "last_name": f"LastName{i}",
                    "age": 20 + (i % 60),
                    "location": {
                        "country": ["USA", "Canada", "UK", "Germany", "France"][i % 5],
                        "city": f"City{i % 100}",
                        "coordinates": {
                            "lat": 40.7128 + (i % 100) * 0.01,
                            "lng": -74.0060 + (i % 100) * 0.01
                        }
                    },
                    "preferences": {
                        "theme": ["dark", "light"][i % 2],
                        "language": ["en", "es", "fr", "de"][i % 4],
                        "notifications": {
                            "email": i % 3 == 0,
                            "push": i % 2 == 0,
                            "sms": i % 5 == 0
                        }
                    }
                },
                "activity": {
                    "last_login": f"2024-01-{(i % 30) + 1:02d}T{(i % 24):02d}:00:00Z",
                    "login_count": i * 3 + 10,
                    "posts_created": i % 50,
                    "comments_made": i % 100
                },
                "settings": {
                    "privacy_level": ["public", "friends", "private"][i % 3],
                    "two_factor_enabled": i % 4 == 0,
                    "marketing_emails": i % 3 != 0
                }
            }
            for i in range(1000)  # 1000 users for demo
        },
        "posts": [
            {
                "id": i,
                "author_id": f"user_{(i % 1000):06d}",
                "title": f"Post Title {i}: {'Long ' * (i % 10)}Discussion About Topic {i % 20}",
                "content": f"This is the content of post {i}. " + 
                          "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * (i % 5 + 1),
                "tags": [f"tag_{j}" for j in range(i % 8)],
                "metadata": {
                    "created_at": f"2024-01-{(i % 30) + 1:02d}T{(i % 24):02d}:{(i % 60):02d}:00Z",
                    "updated_at": f"2024-01-{(i % 30) + 1:02d}T{((i + 1) % 24):02d}:{((i + 30) % 60):02d}:00Z",
                    "view_count": i * 15 + 100,
                    "like_count": i * 3 + 5,
                    "comment_count": i % 25,
                    "share_count": i % 10
                },
                "categories": [
                    "technology", "science", "politics", "entertainment", 
                    "sports", "health", "education", "business"
                ][:(i % 4) + 1],
                "visibility": ["public", "unlisted", "private"][i % 3],
                "featured": i % 20 == 0
            }
            for i in range(2000)  # 2000 posts
        ],
        "analytics": {
            "daily_stats": {
                f"2024-01-{day:02d}": {
                    "active_users": 500 + day * 10 + (day % 7) * 50,
                    "new_posts": 20 + day * 2,
                    "total_views": 10000 + day * 500,
                    "engagement_rate": 0.15 + (day % 10) * 0.01,
                    "bounce_rate": 0.25 - (day % 15) * 0.005,
                    "session_duration_avg": 180 + day * 5,
                    "top_categories": [
                        {"name": "technology", "count": 100 + day * 5},
                        {"name": "science", "count": 80 + day * 3},
                        {"name": "entertainment", "count": 120 + day * 7}
                    ]
                }
                for day in range(1, 32)  # Full month
            },
            "user_segments": {
                "power_users": {
                    "count": 150,
                    "avg_posts_per_day": 5.2,
                    "avg_session_time": 45.5,
                    "retention_rate": 0.95
                },
                "regular_users": {
                    "count": 600,
                    "avg_posts_per_day": 1.8,
                    "avg_session_time": 25.3,
                    "retention_rate": 0.78
                },
                "casual_users": {
                    "count": 250,
                    "avg_posts_per_day": 0.3,
                    "avg_session_time": 12.1,
                    "retention_rate": 0.45
                }
            }
        },
        "configuration": {
            "features": {
                "user_registration": True,
                "email_verification": True,
                "two_factor_auth": True,
                "social_login": {
                    "google": True,
                    "facebook": True,
                    "twitter": True,
                    "github": False
                },
                "content_moderation": {
                    "auto_filter": True,
                    "manual_review": True,
                    "ai_detection": True,
                    "user_reporting": True
                }
            },
            "limits": {
                "max_posts_per_user_per_day": 50,
                "max_post_length": 10000,
                "max_comment_length": 2000,
                "max_file_upload_size_mb": 25,
                "rate_limit_requests_per_minute": 100
            },
            "integrations": {
                "analytics": ["google_analytics", "mixpanel"],
                "cdn": "cloudflare",
                "email_service": "sendgrid",
                "search": "elasticsearch",
                "cache": "redis",
                "database": "postgresql"
            }
        }
    }


async def main():
    """Main demonstration function."""
    # Set up logging for detailed output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Advanced JSON Transformer Demo")
    print("=" * 60)
    
    # Create large, complex dataset
    print("📊 Generating large dataset...")
    sample_data = create_large_dataset()
    json_string = json.dumps(sample_data, separators=(',', ':'))  # Compact format
    
    print(f"📈 Dataset Statistics:")
    print(f"   • Total size: {len(json_string):,} characters ({len(json_string)/1024:.1f} KB)")
    print(f"   • Users: {len(sample_data['users']):,}")
    print(f"   • Posts: {len(sample_data['posts']):,}")
    print(f"   • Analytics entries: {len(sample_data['analytics']['daily_stats']):,}")
    print()
    
    # Create advanced transformer with state-of-the-art settings
    print("⚙️  Initializing Advanced JSON Transformer...")
    transformer = JSONTransformer(
        default_max_size=92160,  # 90KB optimal for LLM processing
        enable_parallel_processing=True,
        max_workers=4,  # Use 4 worker threads
        enable_compression=True,
        memory_limit_mb=2048  # 2GB memory limit
    )
    
    print(f"   • Chunk size: 90KB (92,160 bytes)")
    print(f"   • Parallel processing: Enabled (4 workers)")
    print(f"   • Compression optimization: Enabled")
    print(f"   • Memory limit: 2GB")
    print()
    
    # Use temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Output directory: {temp_dir}")
        print()
        
        try:
            print("🔄 Starting advanced unflatten operation...")
            print("   This will demonstrate:")
            print("   • Intelligent sharding with 90KB chunks")
            print("   • Performance profiling and optimization")
            print("   • Memory-efficient processing")
            print("   • Compression optimizations")
            print()
            
            # Perform the unflatten operation
            result = await transformer.unflatten(
                json_string=json_string,
                max_size=92160,  # 90KB chunks
                output_dir=temp_dir
            )
            
            if result.success:
                print("✅ Advanced Processing Complete!")
                print()
                print(f"📊 Results Summary:")
                print(f"   • Files created: {result.file_count:,}")
                print(f"   • Total output size: {result.total_size:,} bytes ({result.total_size/1024:.1f} KB)")
                print(f"   • Compression ratio: {result.total_size/len(json_string):.3f}")
                print(f"   • Average file size: {result.total_size/result.file_count/1024:.1f} KB")
                print(f"   • Storage efficiency: {(result.total_size/result.file_count)/92160*100:.1f}% of max chunk size")
                print()
                
                # Analyze created files
                output_path = Path(result.output_directory)
                json_files = sorted([f for f in output_path.glob("*.json") if not f.name.startswith('_')])
                
                print(f"📋 File Analysis:")
                print(f"   • Shard files: {len(json_files)}")
                
                # Show size distribution
                sizes = [f.stat().st_size for f in json_files]
                if sizes:
                    print(f"   • Size range: {min(sizes)/1024:.1f} - {max(sizes)/1024:.1f} KB")
                    print(f"   • Average size: {sum(sizes)/len(sizes)/1024:.1f} KB")
                    print(f"   • Size std dev: {(sum((s-sum(sizes)/len(sizes))**2 for s in sizes)/len(sizes))**0.5/1024:.1f} KB")
                
                # Show sample files
                print(f"\n📄 Sample Files:")
                for i, file_path in enumerate(json_files[:3]):
                    file_size = file_path.stat().st_size
                    print(f"   • {file_path.name} ({file_size/1024:.1f} KB)")
                
                if len(json_files) > 3:
                    print(f"   • ... and {len(json_files)-3} more files")
                
                # Show index file info
                index_file = output_path / "_index.json"
                if index_file.exists():
                    print(f"\n🗂️  Index File:")
                    with open(index_file, 'r', encoding='utf-8') as f:
                        index_content = json.load(f)
                        print(f"   • Total shards indexed: {len(index_content.get('shards', []))}")
                        print(f"   • Root shard: {index_content.get('root_shard', 'N/A')}")
                        print(f"   • Data structure: {index_content.get('data_structure', 'N/A')}")
                        print(f"   • Created: {index_content.get('created_at', 'N/A')}")
                
                # Show sample shard content
                if json_files:
                    print(f"\n🔍 Sample Shard Content ({json_files[0].name}):")
                    with open(json_files[0], 'r', encoding='utf-8') as f:
                        shard_content = json.load(f)
                        metadata = shard_content.get('_metadata', {})
                        data_preview = str(shard_content.get('data', {}))[:200]
                        
                        print(f"   • Shard ID: {metadata.get('shardId', 'N/A')}")
                        print(f"   • Data type: {metadata.get('dataType', 'N/A')}")
                        print(f"   • Original path: {metadata.get('originalPath', [])}")
                        print(f"   • Child shards: {len(metadata.get('childIds', []))}")
                        print(f"   • Data preview: {data_preview}...")
                
                # Performance recommendations
                if hasattr(transformer, 'profiler'):
                    print(f"\n🎯 Performance Analysis:")
                    perf_summary = transformer.profiler.get_performance_summary()
                    if perf_summary.get('total_operations', 0) > 0:
                        print(f"   • Total operations: {perf_summary['total_operations']}")
                        print(f"   • Average throughput: {perf_summary['average_throughput_mbps']:.2f} MB/s")
                        print(f"   • Peak memory usage: {perf_summary['average_memory_peak_mb']:.1f} MB")
                        print(f"   • Average CPU usage: {perf_summary['average_cpu_percent']:.1f}%")
                    
                    # Get optimization recommendations
                    recommendations = transformer.profiler.optimize_based_on_history()
                    if recommendations.get("recommendations"):
                        print(f"\n💡 Optimization Recommendations:")
                        for i, rec in enumerate(recommendations["recommendations"][:3], 1):
                            print(f"   {i}. {rec}")
                
                print(f"\n🎉 Demo completed successfully!")
                print(f"   The 90KB chunk size provides optimal balance between:")
                print(f"   • LLM context window efficiency")
                print(f"   • Processing performance")
                print(f"   • Memory usage")
                print(f"   • File system overhead")
                
            else:
                print("❌ Processing failed:")
                if result.errors:
                    for error in result.errors:
                        print(f"   • {error}")
        
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())