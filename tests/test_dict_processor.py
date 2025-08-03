"""Tests for dictionary processor."""

import pytest
from json_transformer.processors.dict_processor import DictProcessor
from json_transformer.types import DataStructure, DataType


class TestDictProcessor:
    """Tests for DictProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DictProcessor()
    
    def test_process_simple_dict(self):
        """Test processing simple dictionary."""
        data = {
            "user1": {"name": "Alice", "age": 30},
            "user2": {"name": "Bob", "age": 25}
        }
        
        result = self.processor.process(data, 25000)
        
        assert result.shards
        assert result.root_shard_id
        assert result.metadata.data_structure == DataStructure.DICT_OF_DICTS
        assert result.metadata.shard_count > 0
    
    def test_process_non_dict_data(self):
        """Test processing non-dictionary data (should fail)."""
        data = [1, 2, 3, 4, 5]
        
        with pytest.raises(ValueError, match="DictProcessor expects dict data"):
            self.processor.process(data, 25000)
    
    def test_process_nested_dict(self):
        """Test processing nested dictionary structure."""
        data = {
            "users": {
                "user1": {
                    "profile": {"name": "Alice", "age": 30},
                    "settings": {"theme": "dark", "notifications": True}
                },
                "user2": {
                    "profile": {"name": "Bob", "age": 25},
                    "settings": {"theme": "light", "notifications": False}
                }
            },
            "config": {
                "version": "1.0",
                "debug": False
            }
        }
        
        result = self.processor.process(data, 25000)
        
        assert len(result.shards) > 1  # Should create multiple shards for nested structure
        
        # Check that we have hierarchical relationships
        root_shard = next(s for s in result.shards if s.id == result.root_shard_id)
        assert len(root_shard.metadata.child_ids) > 0
    
    def test_process_dict_with_lists(self):
        """Test processing dictionary containing lists."""
        data = {
            "users": {
                "user1": {"name": "Alice", "age": 30}
            },
            "items": [
                {"id": 1, "name": "Item 1"},
                {"id": 2, "name": "Item 2"},
                {"id": 3, "name": "Item 3"}
            ],
            "config": {"version": "1.0"}
        }
        
        result = self.processor.process(data, 25000)
        
        assert len(result.shards) > 1
        
        # Should have both dict and list shards
        data_types = [shard.metadata.data_type for shard in result.shards]
        assert DataType.DICT in data_types
        assert DataType.LIST in data_types
    
    def test_process_large_dict_splitting(self):
        """Test processing large dictionary that needs splitting."""
        # Create large dictionary
        data = {f"section_{i}": {f"key_{j}": f"value_{j}" * 100 for j in range(20)} for i in range(10)}
        
        result = self.processor.process(data, 2000)  # Small max size to force splitting
        
        assert len(result.shards) > 1
        
        # Verify all shards are within reasonable size limits
        for shard in result.shards:
            # Allow some overhead, but should be roughly within limits
            assert shard.size < 5000  # Allow some overhead
    
    def test_process_dict_level_simple(self):
        """Test processing single dictionary level."""
        data = {"key1": "value1", "key2": "value2"}
        
        shard = self.processor._process_dict_level(data, 25000, [], None)
        
        assert shard.id.startswith("shard_")
        assert shard.data == data
        assert shard.metadata.data_type == DataType.DICT
        assert shard.metadata.parent_id is None
        assert len(shard.metadata.child_ids) == 0
    
    def test_process_dict_level_with_nested_dict(self):
        """Test processing dictionary level with nested dictionaries."""
        data = {
            "simple_key": "simple_value",
            "nested_dict": {"inner_key": "inner_value"}
        }
        
        shard = self.processor._process_dict_level(data, 25000, [], None)
        
        assert len(shard.metadata.child_ids) == 1  # One child for nested dict
        
        # Check that nested dict is referenced
        assert "nested_dict" in shard.data
        assert "_shard_ref" in shard.data["nested_dict"]
        assert shard.data["nested_dict"]["_shard_type"] == "dict"
    
    def test_process_list_in_dict(self):
        """Test processing list within dictionary."""
        data = [{"item": 1}, {"item": 2}, {"item": 3}]
        original_path = ["items"]
        parent_id = "parent_shard_001"
        
        shards = self.processor._process_list_in_dict(data, 25000, original_path, parent_id)
        
        assert len(shards) >= 1
        
        for shard in shards:
            assert shard.metadata.data_type == DataType.LIST
            assert shard.metadata.parent_id == parent_id
            assert shard.metadata.original_path == original_path
    
    def test_split_current_level_no_splitting_needed(self):
        """Test splitting current level when no splitting is needed."""
        data = {"key1": "value1", "key2": "value2"}
        
        remaining_data, additional_shards = self.processor._split_current_level(
            data, 25000, [], "base_shard"
        )
        
        assert remaining_data == data
        assert len(additional_shards) == 0
    
    def test_split_current_level_splitting_needed(self):
        """Test splitting current level when splitting is needed."""
        # Create large data that needs splitting
        data = {f"key_{i}": f"value_{i}" * 200 for i in range(50)}
        
        remaining_data, additional_shards = self.processor._split_current_level(
            data, 1000, [], "base_shard"  # Small max size to force splitting
        )
        
        assert len(additional_shards) > 0
        assert isinstance(remaining_data, dict)
        
        # Check additional shards
        for shard in additional_shards:
            assert shard.id.startswith("base_shard_overflow_")
            assert shard.metadata.parent_id == "base_shard"
    
    def test_collect_shards_recursive(self):
        """Test recursive shard collection."""
        # Create a simple shard structure
        data = {"users": {"user1": {"name": "Alice"}}}
        result = self.processor.process(data, 25000)
        
        # Manually collect shards to test the method
        all_shards = []
        root_shard = next(s for s in result.shards if s.id == result.root_shard_id)
        self.processor._collect_shards_recursive(root_shard, all_shards)
        
        assert len(all_shards) >= 1
        assert root_shard in all_shards
    
    def test_create_shard_references(self):
        """Test creating shard reference map."""
        data = {
            "users": {"user1": {"name": "Alice"}},
            "config": {"version": "1.0"}
        }
        
        result = self.processor.process(data, 25000)
        references = self.processor.create_shard_references(result.shards)
        
        assert len(references) == len(result.shards)
        
        for shard_id, ref_info in references.items():
            assert "path" in ref_info
            assert "type" in ref_info
            assert "parent" in ref_info
            assert "children" in ref_info
            assert "size" in ref_info
    
    def test_validate_hierarchical_structure_valid(self):
        """Test validation of valid hierarchical structure."""
        data = {"users": {"user1": {"name": "Alice"}}}
        result = self.processor.process(data, 25000)
        
        is_valid = self.processor.validate_hierarchical_structure(result.shards)
        
        assert is_valid
    
    def test_validate_hierarchical_structure_invalid(self):
        """Test validation of invalid hierarchical structure."""
        data = {"users": {"user1": {"name": "Alice"}}}
        result = self.processor.process(data, 25000)
        
        # Break the structure by removing a child shard
        if len(result.shards) > 1:
            # Remove a non-root shard
            non_root_shards = [s for s in result.shards if s.id != result.root_shard_id]
            if non_root_shards:
                result.shards.remove(non_root_shards[0])
                
                is_valid = self.processor.validate_hierarchical_structure(result.shards)
                assert not is_valid
    
    def test_optimize_dictionary_sharding_simple(self):
        """Test optimization analysis for simple dictionary."""
        data = {"key1": "value1", "key2": "value2"}
        
        optimization = self.processor.optimize_dictionary_sharding(data, 25000)
        
        assert "total_size" in optimization
        assert "estimated_shards" in optimization
        assert "max_nesting_level" in optimization
        assert "large_keys" in optimization
        assert "key_count" in optimization
        assert "optimization_suggestions" in optimization
        
        assert optimization["key_count"] == 2
        assert optimization["max_nesting_level"] >= 0
    
    def test_optimize_dictionary_sharding_large_values(self):
        """Test optimization analysis with large values."""
        # Create dictionary with some large values
        large_value = "x" * 20000  # 20KB value
        data = {
            "small_key": "small_value",
            "large_key": large_value
        }
        
        optimization = self.processor.optimize_dictionary_sharding(data, 25000)
        
        assert len(optimization["large_keys"]) > 0
        assert any(s["type"] == "split_large_values" for s in optimization["optimization_suggestions"])
    
    def test_optimize_dictionary_sharding_deep_nesting(self):
        """Test optimization analysis with deep nesting."""
        # Create deeply nested structure
        data = {"level1": {"level2": {"level3": {"level4": {"level5": {"level6": "deep_value"}}}}}}
        
        optimization = self.processor.optimize_dictionary_sharding(data, 25000)
        
        assert optimization["max_nesting_level"] > 5
        assert any(s["type"] == "flatten_deep_nesting" for s in optimization["optimization_suggestions"])
    
    def test_optimize_dictionary_sharding_many_keys(self):
        """Test optimization analysis with many keys."""
        # Create dictionary with many keys
        data = {f"key_{i}": f"value_{i}" for i in range(1500)}
        
        optimization = self.processor.optimize_dictionary_sharding(data, 25000)
        
        assert optimization["key_count"] == 1500
        assert any(s["type"] == "group_related_keys" for s in optimization["optimization_suggestions"])
    
    def test_calculate_max_nesting_level_simple(self):
        """Test calculation of nesting level for simple structure."""
        data = {"key": "value"}
        level = self.processor._calculate_max_nesting_level(data)
        assert level == 0
    
    def test_calculate_max_nesting_level_nested(self):
        """Test calculation of nesting level for nested structure."""
        data = {"level1": {"level2": {"level3": "value"}}}
        level = self.processor._calculate_max_nesting_level(data)
        assert level == 2  # 0-indexed: level1=0, level2=1, level3=2
    
    def test_process_empty_dict(self):
        """Test processing empty dictionary."""
        data = {}
        
        result = self.processor.process(data, 25000)
        
        assert len(result.shards) == 1  # Should create one shard even for empty dict
        assert result.shards[0].data == {}
    
    def test_process_dict_with_shard_references(self):
        """Test that shard references are created correctly."""
        data = {
            "section1": {"key1": "value1"},
            "section2": {"key2": "value2"}
        }
        
        result = self.processor.process(data, 25000)
        
        # Find root shard
        root_shard = next(s for s in result.shards if s.id == result.root_shard_id)
        
        # Check that nested sections are referenced
        assert "section1" in root_shard.data
        assert "section2" in root_shard.data
        
        # Check reference structure
        if isinstance(root_shard.data["section1"], dict) and "_shard_ref" in root_shard.data["section1"]:
            assert root_shard.data["section1"]["_shard_type"] == "dict"
            assert "shard_" in root_shard.data["section1"]["_shard_ref"]