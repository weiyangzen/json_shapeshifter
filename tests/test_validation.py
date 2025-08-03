"""Tests for validation utilities."""

import pytest
import json
from json_transformer.utils.validation import ValidationUtils
from json_transformer.types import DataType, ErrorType
from json_transformer.models import ShardMetadata, FileShard


class TestValidationUtils:
    """Tests for ValidationUtils class."""
    
    def test_validate_valid_json(self):
        """Test validation of valid JSON string."""
        json_string = '{"users": {"user1": {"name": "Alice"}}}'
        result = ValidationUtils.validate_json_string(json_string)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_empty_json(self):
        """Test validation of empty JSON string."""
        result = ValidationUtils.validate_json_string("")
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].type == ErrorType.SYNTAX
        assert "empty" in result.errors[0].message.lower()
    
    def test_validate_invalid_json_syntax(self):
        """Test validation of invalid JSON syntax."""
        json_string = '{"users": {"user1": {"name": "Alice"}'  # Missing closing braces
        result = ValidationUtils.validate_json_string(json_string)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].type == ErrorType.SYNTAX
        assert "syntax" in result.errors[0].message.lower()
    
    def test_validate_unsupported_root_type(self):
        """Test validation of unsupported root type."""
        json_string = '"just a string"'
        result = ValidationUtils.validate_json_string(json_string)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].type == ErrorType.STRUCTURE
        assert "Root element must be dict or list" in result.errors[0].message
    
    def test_validate_deep_nesting_warning(self):
        """Test warning for deep nesting."""
        # Create deeply nested structure
        data = {}
        current = data
        for i in range(25):  # Create 25 levels of nesting
            current[f"level_{i}"] = {}
            current = current[f"level_{i}"]
        
        json_string = json.dumps(data)
        result = ValidationUtils.validate_json_string(json_string)
        
        assert result.is_valid  # Should still be valid
        assert len(result.warnings) > 0
        assert "Deep nesting detected" in result.warnings[0]
    
    def test_circular_reference_detection(self):
        """Test detection of circular references."""
        # Create structure with circular reference
        data = {"a": {}, "b": {}}
        data["a"]["ref"] = data["b"]
        data["b"]["ref"] = data["a"]
        
        has_circular = ValidationUtils._has_circular_references(data)
        assert has_circular
    
    def test_max_depth_calculation(self):
        """Test maximum depth calculation."""
        # Simple nested structure
        data = {
            "level1": {
                "level2": {
                    "level3": "value"
                }
            }
        }
        
        depth = ValidationUtils._calculate_max_depth(data)
        assert depth == 3
        
        # List structure
        list_data = [[[["deep_value"]]]]
        depth = ValidationUtils._calculate_max_depth(list_data)
        assert depth == 4
    
    def test_validate_shard_metadata_valid(self):
        """Test validation of valid shard metadata."""
        metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id="shard_000",
            child_ids=["shard_002"],
            data_type=DataType.DICT,
            original_path=["users"]
        )
        
        result = ValidationUtils.validate_shard_metadata(metadata)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_shard_metadata_self_parent(self):
        """Test validation of shard that is its own parent."""
        metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id="shard_001",  # Self as parent
            child_ids=[],
            data_type=DataType.DICT,
            original_path=[]
        )
        
        result = ValidationUtils.validate_shard_metadata(metadata)
        
        assert not result.is_valid
        assert any("own parent" in error.message for error in result.errors)
    
    def test_validate_shard_metadata_self_child(self):
        """Test validation of shard that is its own child."""
        metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id=None,
            child_ids=["shard_001"],  # Self as child
            data_type=DataType.DICT,
            original_path=[]
        )
        
        result = ValidationUtils.validate_shard_metadata(metadata)
        
        assert not result.is_valid
        assert any("own child" in error.message for error in result.errors)
    
    def test_validate_shard_links_valid(self):
        """Test validation of valid shard links."""
        # Create parent shard
        parent_metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id=None,
            child_ids=["shard_002"],
            data_type=DataType.DICT,
            original_path=["root"]
        )
        parent_shard = FileShard(
            id="shard_001",
            data={"parent": "data"},
            metadata=parent_metadata,
            size=100
        )
        
        # Create child shard
        child_metadata = ShardMetadata(
            shard_id="shard_002",
            parent_id="shard_001",
            child_ids=[],
            data_type=DataType.DICT,
            original_path=["root", "child"]
        )
        child_shard = FileShard(
            id="shard_002",
            data={"child": "data"},
            metadata=child_metadata,
            size=100
        )
        
        result = ValidationUtils.validate_shard_links([parent_shard, child_shard])
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_shard_links_missing_parent(self):
        """Test validation with missing parent shard."""
        metadata = ShardMetadata(
            shard_id="shard_002",
            parent_id="shard_001",  # Parent doesn't exist
            child_ids=[],
            data_type=DataType.DICT,
            original_path=[]
        )
        shard = FileShard(
            id="shard_002",
            data={},
            metadata=metadata,
            size=100
        )
        
        result = ValidationUtils.validate_shard_links([shard])
        
        assert not result.is_valid
        assert any("Parent shard 'shard_001' not found" in error.message for error in result.errors)
    
    def test_validate_shard_links_missing_child(self):
        """Test validation with missing child shard."""
        metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id=None,
            child_ids=["shard_002"],  # Child doesn't exist
            data_type=DataType.DICT,
            original_path=[]
        )
        shard = FileShard(
            id="shard_001",
            data={},
            metadata=metadata,
            size=100
        )
        
        result = ValidationUtils.validate_shard_links([shard])
        
        assert not result.is_valid
        assert any("Child shard 'shard_002' not found" in error.message for error in result.errors)
    
    def test_validate_shard_links_broken_bidirectional(self):
        """Test validation with broken bidirectional links."""
        # Parent claims child, but child doesn't claim parent
        parent_metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id=None,
            child_ids=["shard_002"],
            data_type=DataType.DICT,
            original_path=[]
        )
        parent_shard = FileShard(
            id="shard_001",
            data={},
            metadata=parent_metadata,
            size=100
        )
        
        child_metadata = ShardMetadata(
            shard_id="shard_002",
            parent_id="shard_999",  # Wrong parent
            child_ids=[],
            data_type=DataType.DICT,
            original_path=[]
        )
        child_shard = FileShard(
            id="shard_002",
            data={},
            metadata=child_metadata,
            size=100
        )
        
        result = ValidationUtils.validate_shard_links([parent_shard, child_shard])
        
        assert not result.is_valid
        assert any("Bidirectional link broken" in error.message for error in result.errors)
    
    def test_validate_shard_links_no_root(self):
        """Test validation with no root shard."""
        # Create shard with parent (no root)
        metadata = ShardMetadata(
            shard_id="shard_002",
            parent_id="shard_001",
            child_ids=[],
            data_type=DataType.DICT,
            original_path=[]
        )
        shard = FileShard(
            id="shard_002",
            data={},
            metadata=metadata,
            size=100
        )
        
        result = ValidationUtils.validate_shard_links([shard])
        
        assert not result.is_valid
        assert any("No root shard found" in error.message for error in result.errors)
    
    def test_validate_shard_links_multiple_roots(self):
        """Test validation with multiple root shards."""
        # Create two root shards
        root1_metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id=None,
            child_ids=[],
            data_type=DataType.DICT,
            original_path=[]
        )
        root1_shard = FileShard(
            id="shard_001",
            data={},
            metadata=root1_metadata,
            size=100
        )
        
        root2_metadata = ShardMetadata(
            shard_id="shard_002",
            parent_id=None,
            child_ids=[],
            data_type=DataType.DICT,
            original_path=[]
        )
        root2_shard = FileShard(
            id="shard_002",
            data={},
            metadata=root2_metadata,
            size=100
        )
        
        result = ValidationUtils.validate_shard_links([root1_shard, root2_shard])
        
        assert result.is_valid  # Should still be valid
        assert len(result.warnings) > 0
        assert "Multiple root shards found" in result.warnings[0]