"""Tests for mixed structure processor."""

import pytest
from json_transformer.processors.mixed_processor import MixedProcessor
from json_transformer.types import DataStructure, DataType


class TestMixedProcessor:
    """Tests for MixedProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = MixedProcessor()
    
    def test_process_mixed_dict_structure(self):
        """Test processing mixed dictionary structure."""
        data = {
            "metadata": {"version": "1.0", "created": "2024-01-01"},
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"}
            ],
            "config": {
                "settings": {"theme": "dark", "notifications": True},
                "features": ["feature1", "feature2"]
            },
            "simple_value": "test"
        }
        
        result = self.processor.process(data, 25000)
        
        assert result.shards
        assert result.root_shard_id
        assert result.metadata.data_structure == DataStructure.MIXED
        assert result.metadata.shard_count > 0
    
    def test_process_mixed_list_structure(self):
        """Test processing mixed list structure."""
        data = [
            {"type": "dict", "data": {"key": "value"}},
            [1, 2, 3, 4, 5],
            "simple_string",
            {"complex": {"nested": {"structure": "value"}}},
            42
        ]
        
        result = self.processor.process(data, 25000)
        
        assert result.shards
        assert result.root_shard_id
        assert result.metadata.data_structure == DataStructure.MIXED
        assert result.metadata.shard_count > 0
    
    def test_process_unsupported_root_type(self):
        """Test processing unsupported root type."""
        data = "simple_string"
        
        with pytest.raises(ValueError, match="Unsupported root data type"):
            self.processor.process(data, 25000)
    
    def test_analyze_mixed_structure_dict(self):
        """Test analysis of mixed dictionary structure."""
        data = {
            "dict_value": {"nested": "dict"},
            "list_value": [1, 2, 3],
            "primitive_value": "string"
        }
        
        analysis = self.processor._analyze_mixed_structure(data)
        
        assert analysis["root_type"] == "dict"
        assert analysis["dict_count"] >= 2  # root + nested dict
        assert analysis["list_count"] >= 1
        assert analysis["primitive_count"] >= 1
        assert analysis["complexity_score"] > 0
    
    def test_analyze_mixed_structure_list(self):
        """Test analysis of mixed list structure."""
        data = [
            {"dict": "item"},
            [1, 2, 3],
            "primitive"
        ]
        
        analysis = self.processor._analyze_mixed_structure(data)
        
        assert analysis["root_type"] == "list"
        assert analysis["dict_count"] >= 1
        assert analysis["list_count"] >= 2  # root + nested list
        assert analysis["primitive_count"] >= 1
    
    def test_calculate_complexity_score_simple(self):
        """Test complexity score calculation for simple structure."""
        analysis = {
            "dict_count": 1,
            "list_count": 0,
            "primitive_count": 2,
            "max_depth": 1,
            "structure_map": {
                "root": {"type": "dict", "depth": 0},
                "key1": {"type": "primitive", "depth": 1},
                "key2": {"type": "primitive", "depth": 1}
            }
        }
        
        score = self.processor._calculate_complexity_score(analysis)
        
        assert score >= 0.0
        assert score <= 10.0
    
    def test_calculate_complexity_score_complex(self):
        """Test complexity score calculation for complex structure."""
        analysis = {
            "dict_count": 10,
            "list_count": 5,
            "primitive_count": 1500,  # Large number
            "max_depth": 8,  # Deep nesting
            "structure_map": {
                "root": {"type": "dict", "depth": 0},
                "level1": {"type": "dict", "depth": 1},
                "level1.nested": {"type": "list", "depth": 2},
                "level1.other": {"type": "primitive", "depth": 2}
            }
        }
        
        score = self.processor._calculate_complexity_score(analysis)
        
        assert score > 3.0  # Should be high complexity
        assert score <= 10.0
    
    def test_process_mixed_dict_with_references(self):
        """Test processing mixed dict creates proper references."""
        data = {
            "simple": "value",
            "nested_dict": {"key": "value"},
            "nested_list": [1, 2, 3]
        }
        
        result = self.processor.process(data, 25000)
        
        # Find root shard
        root_shard = next(s for s in result.shards if s.id == result.root_shard_id)
        
        # Check that references are created
        assert "simple" in root_shard.data
        assert root_shard.data["simple"] == "value"
        
        # Check references to nested structures
        if "nested_dict" in root_shard.data and isinstance(root_shard.data["nested_dict"], dict):
            if "_shard_ref" in root_shard.data["nested_dict"]:
                assert root_shard.data["nested_dict"]["_shard_type"] == "dict"
                assert root_shard.data["nested_dict"]["_processor"] == "dict_processor"
        
        if "nested_list" in root_shard.data and isinstance(root_shard.data["nested_list"], dict):
            if "_shard_ref" in root_shard.data["nested_list"]:
                assert root_shard.data["nested_list"]["_shard_type"] == "list"
                assert root_shard.data["nested_list"]["_processor"] == "list_processor"
    
    def test_should_process_as_dict_large_item(self):
        """Test decision to process large dict item separately."""
        # Create large dictionary
        large_dict = {f"key_{i}": f"value_{i}" * 100 for i in range(50)}
        
        should_process = self.processor._should_process_as_dict(large_dict, 25000)
        
        assert should_process
    
    def test_should_process_as_dict_many_keys(self):
        """Test decision to process dict with many keys separately."""
        many_keys_dict = {f"key_{i}": f"value_{i}" for i in range(20)}
        
        should_process = self.processor._should_process_as_dict(many_keys_dict, 25000)
        
        assert should_process
    
    def test_should_process_as_dict_nested_structure(self):
        """Test decision to process dict with nested structures separately."""
        nested_dict = {
            "simple": "value",
            "nested": {"inner": "value"}
        }
        
        should_process = self.processor._should_process_as_dict(nested_dict, 25000)
        
        assert should_process
    
    def test_should_process_as_dict_simple_item(self):
        """Test decision not to process simple dict separately."""
        simple_dict = {"key1": "value1", "key2": "value2"}
        
        should_process = self.processor._should_process_as_dict(simple_dict, 25000)
        
        assert not should_process
    
    def test_should_process_as_list_large_item(self):
        """Test decision to process large list item separately."""
        # Create large list
        large_list = [{"item": f"data_{i}" * 100} for i in range(50)]
        
        should_process = self.processor._should_process_as_list(large_list, 25000)
        
        assert should_process
    
    def test_should_process_as_list_many_elements(self):
        """Test decision to process list with many elements separately."""
        many_elements_list = [f"item_{i}" for i in range(30)]
        
        should_process = self.processor._should_process_as_list(many_elements_list, 25000)
        
        assert should_process
    
    def test_should_process_as_list_nested_structure(self):
        """Test decision to process list with nested structures separately."""
        nested_list = [
            "simple",
            {"nested": "dict"},
            [1, 2, 3]
        ]
        
        should_process = self.processor._should_process_as_list(nested_list, 25000)
        
        assert should_process
    
    def test_should_process_as_list_simple_item(self):
        """Test decision not to process simple list separately."""
        simple_list = [1, 2, 3, 4, 5]
        
        should_process = self.processor._should_process_as_list(simple_list, 25000)
        
        assert not should_process
    
    def test_create_wrapper_shard(self):
        """Test creation of wrapper shard."""
        data = {
            "_list_item_index": 5,
            "_shard_ref": "shard_123",
            "_shard_type": "dict",
            "_processor": "dict_processor"
        }
        
        wrapper_shard = self.processor._create_wrapper_shard(data, 5, DataType.DICT)
        
        assert wrapper_shard.data == data
        assert wrapper_shard.metadata.data_type == DataType.DICT
        assert wrapper_shard.metadata.original_path == ["[5]"]
    
    def test_process_mixed_list_with_complex_items(self):
        """Test processing mixed list with items that need separate processing."""
        # Create list with items that should be processed separately
        large_dict = {f"key_{i}": f"value_{i}" * 50 for i in range(20)}
        large_list = [{"item": f"data_{i}" * 30} for i in range(25)]
        
        data = [
            "simple_item",
            large_dict,  # Should be processed separately
            large_list,  # Should be processed separately
            {"small": "dict"},  # Should stay in main list
            42
        ]
        
        result = self.processor.process(data, 5000)  # Small max size to force processing
        
        assert len(result.shards) > 1
        
        # Should have wrapper shards for complex items
        wrapper_shards = [s for s in result.shards 
                         if isinstance(s.data, dict) and "_list_item_index" in s.data]
        assert len(wrapper_shards) >= 2  # At least for large_dict and large_list
    
    def test_get_mixed_structure_statistics_empty(self):
        """Test getting statistics for empty shard list."""
        stats = self.processor.get_mixed_structure_statistics([])
        
        assert stats["total_shards"] == 0
        assert stats["processor_distribution"] == {"dict_processor": 0, "list_processor": 0, "mixed_processor": 0}
        assert stats["data_type_distribution"] == {}
    
    def test_get_mixed_structure_statistics_with_shards(self):
        """Test getting statistics for shard list."""
        data = {
            "dict_section": {"key": "value"},
            "list_section": [1, 2, 3],
            "simple": "value"
        }
        
        result = self.processor.process(data, 25000)
        stats = self.processor.get_mixed_structure_statistics(result.shards)
        
        assert stats["total_shards"] == len(result.shards)
        assert "processor_distribution" in stats
        assert "data_type_distribution" in stats
        assert "complexity_metrics" in stats
        
        # Check complexity metrics
        assert "total_size" in stats["complexity_metrics"]
        assert "average_shard_size" in stats["complexity_metrics"]
        assert "structure_diversity" in stats["complexity_metrics"]
    
    def test_validate_mixed_structure_integrity_valid(self):
        """Test validation of valid mixed structure."""
        data = {
            "dict_section": {"key": "value"},
            "list_section": [1, 2, 3]
        }
        
        result = self.processor.process(data, 25000)
        validation = self.processor.validate_mixed_structure_integrity(result.shards)
        
        assert validation["is_valid"]
        assert len(validation["errors"]) == 0
    
    def test_validate_mixed_structure_integrity_empty(self):
        """Test validation of empty shard list."""
        validation = self.processor.validate_mixed_structure_integrity([])
        
        assert validation["is_valid"]
        assert len(validation["warnings"]) == 1
        assert "No shards to validate" in validation["warnings"][0]
    
    def test_validate_mixed_structure_integrity_broken_reference(self):
        """Test validation with broken reference."""
        data = {"simple": "value"}
        result = self.processor.process(data, 25000)
        
        # Break a reference by modifying shard data
        if result.shards:
            result.shards[0].data["broken_ref"] = {
                "_shard_ref": "nonexistent_shard_id",
                "_shard_type": "dict"
            }
            
            validation = self.processor.validate_mixed_structure_integrity(result.shards)
            
            assert not validation["is_valid"]
            assert len(validation["errors"]) > 0
            assert any("Broken reference" in error for error in validation["errors"])
    
    def test_analyze_structure_recursive_deep_nesting(self):
        """Test recursive structure analysis with deep nesting."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": ["deep", "list"]
                    }
                }
            }
        }
        
        analysis = self.processor._analyze_mixed_structure(data)
        
        assert analysis["max_depth"] >= 4
        assert analysis["dict_count"] >= 4  # level1, level2, level3, level4 parent
        assert analysis["list_count"] >= 1
    
    def test_process_mixed_structure_with_errors(self):
        """Test processing mixed structure with items that cause errors."""
        # Create data that might cause processing errors
        class NonSerializable:
            pass
        
        data = {
            "valid_dict": {"key": "value"},
            "valid_list": [1, 2, 3],
            "problematic": NonSerializable()  # This might cause issues
        }
        
        # Should not crash, but may include error markers
        result = self.processor.process(data, 25000)
        
        assert result.shards
        assert result.root_shard_id
    
    def test_setup_list_sequential_linking(self):
        """Test setting up sequential linking for list shards."""
        # Create some test list shards
        data = [1, 2, 3, 4, 5]
        result = self.processor.list_processor.process(data, 25000)
        
        # Get list shards
        list_shards = [s for s in result.shards if s.metadata.data_type == DataType.LIST]
        
        if len(list_shards) > 1:
            # Clear existing links
            for shard in list_shards:
                shard.metadata.next_shard = None
                shard.metadata.previous_shard = None
            
            # Set up linking
            self.processor._setup_list_sequential_linking(list_shards)
            
            # Verify linking
            for i in range(len(list_shards)):
                if i > 0:
                    assert list_shards[i].metadata.previous_shard == list_shards[i-1].id
                if i < len(list_shards) - 1:
                    assert list_shards[i].metadata.next_shard == list_shards[i+1].id
    
    def test_process_empty_mixed_structures(self):
        """Test processing empty mixed structures."""
        # Empty dict
        result_dict = self.processor.process({}, 25000)
        assert len(result_dict.shards) >= 1
        
        # Empty list
        result_list = self.processor.process([], 25000)
        assert len(result_list.shards) >= 1