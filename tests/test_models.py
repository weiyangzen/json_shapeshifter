"""Tests for data models."""

import pytest
from datetime import datetime
from json_transformer.types import DataType, DataStructure
from json_transformer.models import ShardMetadata, GlobalMetadata, FileShard


class TestShardMetadata:
    """Tests for ShardMetadata class."""
    
    def test_create_valid_metadata(self):
        """Test creating valid shard metadata."""
        metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id="shard_000",
            child_ids=["shard_002", "shard_003"],
            data_type=DataType.DICT,
            original_path=["users", "profiles"]
        )
        
        assert metadata.shard_id == "shard_001"
        assert metadata.parent_id == "shard_000"
        assert metadata.child_ids == ["shard_002", "shard_003"]
        assert metadata.data_type == DataType.DICT
        assert metadata.original_path == ["users", "profiles"]
        assert metadata.version == "1.0.0"
    
    def test_invalid_shard_id(self):
        """Test validation of invalid shard ID."""
        with pytest.raises(ValueError, match="shard_id must start with 'shard_'"):
            ShardMetadata(
                shard_id="invalid_001",
                parent_id=None,
                child_ids=[],
                data_type=DataType.DICT,
                original_path=[]
            )
    
    def test_empty_shard_id(self):
        """Test validation of empty shard ID."""
        with pytest.raises(ValueError, match="shard_id cannot be empty"):
            ShardMetadata(
                shard_id="",
                parent_id=None,
                child_ids=[],
                data_type=DataType.DICT,
                original_path=[]
            )
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id="shard_000",
            child_ids=["shard_002"],
            data_type=DataType.LIST,
            original_path=["data", "items"],
            next_shard="shard_002"
        )
        
        result = metadata.to_dict()
        expected = {
            "shardId": "shard_001",
            "parentId": "shard_000",
            "childIds": ["shard_002"],
            "dataType": "list",
            "originalPath": ["data", "items"],
            "nextShard": "shard_002",
            "previousShard": None,
            "version": "1.0.0"
        }
        
        assert result == expected
    
    def test_from_dict_creation(self):
        """Test creation from dictionary."""
        data = {
            "shardId": "shard_001",
            "parentId": "shard_000",
            "childIds": ["shard_002"],
            "dataType": "dict",
            "originalPath": ["users"],
            "version": "1.0.0"
        }
        
        metadata = ShardMetadata.from_dict(data)
        
        assert metadata.shard_id == "shard_001"
        assert metadata.parent_id == "shard_000"
        assert metadata.child_ids == ["shard_002"]
        assert metadata.data_type == DataType.DICT
        assert metadata.original_path == ["users"]
    
    def test_child_management(self):
        """Test adding and removing children."""
        metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id=None,
            child_ids=[],
            data_type=DataType.DICT,
            original_path=[]
        )
        
        # Add child
        metadata.add_child("shard_002")
        assert "shard_002" in metadata.child_ids
        
        # Add duplicate child (should not duplicate)
        metadata.add_child("shard_002")
        assert metadata.child_ids.count("shard_002") == 1
        
        # Remove child
        metadata.remove_child("shard_002")
        assert "shard_002" not in metadata.child_ids
    
    def test_utility_methods(self):
        """Test utility methods."""
        # Root shard
        root_metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id=None,
            child_ids=["shard_002"],
            data_type=DataType.DICT,
            original_path=["root"]
        )
        
        assert root_metadata.is_root()
        assert not root_metadata.is_leaf()
        assert root_metadata.has_children()
        
        # Leaf shard
        leaf_metadata = ShardMetadata(
            shard_id="shard_002",
            parent_id="shard_001",
            child_ids=[],
            data_type=DataType.DICT,
            original_path=["root", "child"]
        )
        
        assert not leaf_metadata.is_root()
        assert leaf_metadata.is_leaf()
        assert not leaf_metadata.has_children()
    
    def test_path_string(self):
        """Test path string generation."""
        metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id=None,
            child_ids=[],
            data_type=DataType.DICT,
            original_path=["users", "profiles", "settings"]
        )
        
        assert metadata.get_path_string() == "users.profiles.settings"
        
        # Empty path
        empty_metadata = ShardMetadata(
            shard_id="shard_002",
            parent_id=None,
            child_ids=[],
            data_type=DataType.DICT,
            original_path=[]
        )
        
        assert empty_metadata.get_path_string() == "root"


class TestGlobalMetadata:
    """Tests for GlobalMetadata class."""
    
    def test_create_valid_global_metadata(self):
        """Test creating valid global metadata."""
        created_at = datetime.now()
        metadata = GlobalMetadata(
            version="1.0.0",
            original_size=1024000,
            shard_count=5,
            root_shard="shard_001",
            created_at=created_at,
            data_structure=DataStructure.DICT_OF_DICTS
        )
        
        assert metadata.version == "1.0.0"
        assert metadata.original_size == 1024000
        assert metadata.shard_count == 5
        assert metadata.root_shard == "shard_001"
        assert metadata.created_at == created_at
        assert metadata.data_structure == DataStructure.DICT_OF_DICTS
    
    def test_invalid_shard_count(self):
        """Test validation of invalid shard count."""
        with pytest.raises(ValueError, match="shard_count must be positive"):
            GlobalMetadata(
                version="1.0.0",
                original_size=1024,
                shard_count=0,
                root_shard="shard_001",
                created_at=datetime.now(),
                data_structure=DataStructure.LIST
            )
    
    def test_invalid_root_shard(self):
        """Test validation of invalid root shard."""
        with pytest.raises(ValueError, match="root_shard must start with 'shard_'"):
            GlobalMetadata(
                version="1.0.0",
                original_size=1024,
                shard_count=1,
                root_shard="invalid_001",
                created_at=datetime.now(),
                data_structure=DataStructure.LIST
            )
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        created_at = datetime(2024, 1, 1, 12, 0, 0)
        metadata = GlobalMetadata(
            version="1.0.0",
            original_size=2048,
            shard_count=3,
            root_shard="shard_001",
            created_at=created_at,
            data_structure=DataStructure.MIXED
        )
        
        result = metadata.to_dict()
        expected = {
            "version": "1.0.0",
            "originalSize": 2048,
            "shardCount": 3,
            "rootShard": "shard_001",
            "createdAt": "2024-01-01T12:00:00",
            "dataStructure": "mixed"
        }
        
        assert result == expected
    
    def test_utility_methods(self):
        """Test utility methods."""
        metadata = GlobalMetadata(
            version="1.0.0",
            original_size=10485760,  # 10MB
            shard_count=4,
            root_shard="shard_001",
            created_at=datetime.now(),
            data_structure=DataStructure.DICT_OF_DICTS
        )
        
        # Size calculations
        assert metadata.get_size_mb() == 10.0
        assert metadata.get_average_shard_size() == 2621440  # 10MB / 4
        assert metadata.is_large_dataset(threshold_mb=5.0)
        assert not metadata.is_large_dataset(threshold_mb=15.0)


class TestFileShard:
    """Tests for FileShard class."""
    
    def test_create_valid_file_shard(self):
        """Test creating valid file shard."""
        metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id=None,
            child_ids=[],
            data_type=DataType.DICT,
            original_path=["users"]
        )
        
        shard = FileShard(
            id="shard_001",
            data={"user1": {"name": "Alice"}},
            metadata=metadata,
            size=1024
        )
        
        assert shard.id == "shard_001"
        assert shard.data == {"user1": {"name": "Alice"}}
        assert shard.metadata == metadata
        assert shard.size == 1024
    
    def test_id_metadata_mismatch(self):
        """Test validation of ID and metadata mismatch."""
        metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id=None,
            child_ids=[],
            data_type=DataType.DICT,
            original_path=[]
        )
        
        with pytest.raises(ValueError, match="id must match metadata.shard_id"):
            FileShard(
                id="shard_002",  # Different from metadata
                data={},
                metadata=metadata,
                size=100
            )
    
    def test_to_file_schema(self):
        """Test conversion to file schema."""
        metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id=None,
            child_ids=["shard_002"],
            data_type=DataType.DICT,
            original_path=["users"]
        )
        
        shard = FileShard(
            id="shard_001",
            data={"user1": {"name": "Alice"}},
            metadata=metadata,
            size=1024
        )
        
        schema = shard.to_file_schema()
        
        assert "_metadata" in schema
        assert "data" in schema
        assert schema["_metadata"]["shardId"] == "shard_001"
        assert schema["data"] == {"user1": {"name": "Alice"}}
    
    def test_from_file_schema(self):
        """Test creation from file schema."""
        schema = {
            "_metadata": {
                "shardId": "shard_001",
                "parentId": None,
                "childIds": [],
                "dataType": "dict",
                "originalPath": ["users"],
                "version": "1.0.0"
            },
            "data": {"user1": {"name": "Bob"}}
        }
        
        shard = FileShard.from_file_schema(schema)
        
        assert shard.id == "shard_001"
        assert shard.data == {"user1": {"name": "Bob"}}
        assert shard.metadata.shard_id == "shard_001"
        assert shard.metadata.data_type == DataType.DICT
    
    def test_utility_methods(self):
        """Test utility methods."""
        metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id=None,
            child_ids=[],
            data_type=DataType.DICT,
            original_path=[]
        )
        
        # Test with nested data
        nested_shard = FileShard(
            id="shard_001",
            data={"user": {"profile": {"name": "Alice"}}},
            metadata=metadata,
            size=2048
        )
        
        assert nested_shard.get_size_kb() == 2.0
        assert nested_shard.is_oversized(1024)
        assert not nested_shard.is_oversized(4096)
        assert nested_shard.has_nested_data()
        
        # Test with flat data
        flat_shard = FileShard(
            id="shard_001",
            data={"name": "Alice", "age": 30},
            metadata=metadata,
            size=512
        )
        
        assert not flat_shard.has_nested_data()
        assert "dict with 2 keys" in flat_shard.get_data_summary()