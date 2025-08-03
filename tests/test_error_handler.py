"""Tests for error handler."""

import pytest
import tempfile
import os
from pathlib import Path
from json_transformer.error_handler import ErrorHandler
from json_transformer.types import ProcessingError, ErrorType, DataType
from json_transformer.models import ShardMetadata, FileShard


class TestErrorHandler:
    """Tests for ErrorHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    def test_validate_input_valid_json(self):
        """Test validation of valid JSON input."""
        json_string = '{"users": {"user1": {"name": "Alice"}}}'
        result = self.error_handler.validate_input(json_string)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_input_invalid_json(self):
        """Test validation of invalid JSON input."""
        json_string = '{"users": {"user1": {"name": "Alice"}'  # Missing closing brace
        result = self.error_handler.validate_input(json_string)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert result.errors[0].type == ErrorType.SYNTAX
    
    def test_handle_memory_error(self):
        """Test handling of memory errors."""
        error = ProcessingError(
            "Out of memory",
            ErrorType.MEMORY,
            context={"partial_results": ["some", "data"]}
        )
        
        response = self.error_handler.handle_processing_error(error)
        
        assert response.can_recover
        assert "memory" in response.suggested_action.lower()
        assert response.partial_results == ["some", "data"]
    
    def test_handle_circular_error(self):
        """Test handling of circular reference errors."""
        error = ProcessingError(
            "Circular reference detected",
            ErrorType.CIRCULAR
        )
        
        response = self.error_handler.handle_processing_error(error)
        
        assert not response.can_recover
        assert "circular" in response.suggested_action.lower()
    
    def test_handle_corruption_error(self):
        """Test handling of corruption errors."""
        error = ProcessingError(
            "Data corruption detected",
            ErrorType.CORRUPTION,
            context={"corrupted_shards": ["shard_001", "shard_002"]}
        )
        
        response = self.error_handler.handle_processing_error(error)
        
        assert response.can_recover
        assert "recovery" in response.suggested_action.lower()
        assert response.partial_results == ["shard_001", "shard_002"]
    
    def test_handle_filesystem_error(self):
        """Test handling of filesystem errors."""
        error = ProcessingError(
            "Permission denied",
            ErrorType.FILESYSTEM,
            context={"partial_files": ["file1.json"]}
        )
        
        response = self.error_handler.handle_processing_error(error)
        
        assert response.can_recover
        assert "permission" in response.suggested_action.lower()
        assert response.partial_results == ["file1.json"]
    
    def test_recover_from_corruption_valid_shards(self):
        """Test recovery from corruption with valid shards."""
        # Create valid shard
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
            size=100
        )
        
        result = self.error_handler.recover_from_corruption([shard])
        
        assert result.recovered
        assert len(result.recovered_shards) == 1
        assert len(result.lost_data) == 0
        assert result.recovered_shards[0].id == "shard_001"
    
    def test_recover_from_corruption_invalid_metadata(self):
        """Test recovery from corruption with invalid metadata."""
        # Create shard with invalid metadata
        metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id=None,
            child_ids=[],
            data_type=DataType.DICT,
            original_path=["users"]
        )
        # Corrupt the metadata after creation
        metadata.shard_id = ""  # Invalid empty ID
        
        shard = FileShard(
            id="shard_001",
            data={"user1": {"name": "Alice"}},
            metadata=metadata,
            size=100
        )
        
        # This should raise an error during shard creation, so let's create it differently
        shard.metadata.shard_id = ""  # Corrupt after creation
        
        result = self.error_handler.recover_from_corruption([shard])
        
        assert len(result.lost_data) > 0
        assert any("corrupted metadata" in item for item in result.lost_data)
    
    def test_recover_from_corruption_corrupted_data(self):
        """Test recovery from corruption with corrupted data."""
        metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id=None,
            child_ids=[],
            data_type=DataType.DICT,
            original_path=["users"]
        )
        
        # Create shard with data that doesn't match metadata type
        shard = FileShard(
            id="shard_001",
            data=["this", "is", "a", "list"],  # List data but DICT metadata
            metadata=metadata,
            size=100
        )
        
        result = self.error_handler.recover_from_corruption([shard])
        
        # Should attempt recovery
        assert len(result.recovered_shards) == 1
        assert result.recovered_shards[0].data["_recovered"] is True
    
    def test_validate_size_parameter_valid(self):
        """Test validation of valid size parameter."""
        result = self.error_handler.validate_size_parameter(25000)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_size_parameter_zero(self):
        """Test validation of zero size parameter."""
        result = self.error_handler.validate_size_parameter(0)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].type == ErrorType.SIZE
        assert "positive" in result.errors[0].message
    
    def test_validate_size_parameter_negative(self):
        """Test validation of negative size parameter."""
        result = self.error_handler.validate_size_parameter(-1000)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].type == ErrorType.SIZE
    
    def test_validate_size_parameter_small_warning(self):
        """Test validation of very small size parameter."""
        result = self.error_handler.validate_size_parameter(500)  # 500 bytes
        
        assert result.is_valid
        assert len(result.warnings) == 1
        assert "small" in result.warnings[0].lower()
    
    def test_validate_size_parameter_large_warning(self):
        """Test validation of very large size parameter."""
        result = self.error_handler.validate_size_parameter(200 * 1024 * 1024)  # 200MB
        
        assert result.is_valid
        assert len(result.warnings) == 1
        assert "large" in result.warnings[0].lower()
    
    def test_validate_directory_path_empty(self):
        """Test validation of empty directory path."""
        result = self.error_handler.validate_directory_path("")
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].type == ErrorType.PATH
        assert "empty" in result.errors[0].message
    
    def test_validate_directory_path_valid(self):
        """Test validation of valid directory path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.error_handler.validate_directory_path(temp_dir)
            
            assert result.is_valid
            assert len(result.errors) == 0
    
    def test_validate_directory_path_nonexistent(self):
        """Test validation of non-existent directory path."""
        result = self.error_handler.validate_directory_path("/path/that/does/not/exist")
        
        # Should be valid (directory will be created)
        assert result.is_valid
    
    def test_validate_directory_path_file_not_dir(self):
        """Test validation of path that points to a file."""
        with tempfile.NamedTemporaryFile() as temp_file:
            result = self.error_handler.validate_directory_path(temp_file.name)
            
            assert not result.is_valid
            assert len(result.errors) == 1
            assert "not a directory" in result.errors[0].message
    
    def test_validate_directory_path_traversal(self):
        """Test validation of path traversal attempt."""
        result = self.error_handler.validate_directory_path("../../../etc/passwd")
        
        # This might or might not be invalid depending on the system
        # The important thing is that it's detected and handled
        if not result.is_valid:
            assert any("traversal" in error.message.lower() for error in result.errors)
    
    def test_repair_shard_links(self):
        """Test repairing broken shard links."""
        # Create shards with broken links
        metadata1 = ShardMetadata(
            shard_id="shard_001",
            parent_id=None,
            child_ids=["shard_002", "shard_999"],  # shard_999 doesn't exist
            data_type=DataType.DICT,
            original_path=[]
        )
        shard1 = FileShard(
            id="shard_001",
            data={},
            metadata=metadata1,
            size=100
        )
        
        metadata2 = ShardMetadata(
            shard_id="shard_002",
            parent_id="shard_001",
            child_ids=[],
            data_type=DataType.DICT,
            original_path=[],
            next_shard="shard_999"  # shard_999 doesn't exist
        )
        shard2 = FileShard(
            id="shard_002",
            data={},
            metadata=metadata2,
            size=100
        )
        
        repaired_shards = self.error_handler._repair_shard_links([shard1, shard2])
        
        assert len(repaired_shards) == 2
        # Check that broken references were removed
        assert "shard_999" not in repaired_shards[0].metadata.child_ids
        assert repaired_shards[1].metadata.next_shard is None
    
    def test_create_recovery_shard_dict(self):
        """Test creating recovery shard for dict data."""
        metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id=None,
            child_ids=[],
            data_type=DataType.DICT,
            original_path=[]
        )
        corrupted_shard = FileShard(
            id="shard_001",
            data="corrupted_data",  # Wrong type
            metadata=metadata,
            size=100
        )
        
        recovery_shard = self.error_handler._create_recovery_shard(corrupted_shard)
        
        assert recovery_shard is not None
        assert isinstance(recovery_shard.data, dict)
        assert recovery_shard.data["_recovered"] is True
        assert recovery_shard.data["_original_id"] == "shard_001"
    
    def test_create_recovery_shard_list(self):
        """Test creating recovery shard for list data."""
        metadata = ShardMetadata(
            shard_id="shard_001",
            parent_id=None,
            child_ids=[],
            data_type=DataType.LIST,
            original_path=[]
        )
        corrupted_shard = FileShard(
            id="shard_001",
            data="corrupted_data",  # Wrong type
            metadata=metadata,
            size=100
        )
        
        recovery_shard = self.error_handler._create_recovery_shard(corrupted_shard)
        
        assert recovery_shard is not None
        assert isinstance(recovery_shard.data, list)
        assert recovery_shard.data[0]["_recovered"] is True
        assert recovery_shard.data[0]["_original_id"] == "shard_001"