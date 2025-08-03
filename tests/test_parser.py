"""Tests for JSON parser."""

import pytest
import json
from json_transformer.parser import JSONParser
from json_transformer.types import DataStructure, ErrorType


class TestJSONParser:
    """Tests for JSONParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = JSONParser()
    
    def test_parse_valid_dict_of_dicts(self):
        """Test parsing valid dictionary-of-dictionaries JSON."""
        json_string = '''
        {
            "users": {
                "user1": {"name": "Alice", "age": 30},
                "user2": {"name": "Bob", "age": 25}
            },
            "settings": {
                "theme": "dark",
                "notifications": true
            }
        }
        '''
        
        data, structure_type = self.parser.parse(json_string)
        
        assert isinstance(data, dict)
        assert structure_type == DataStructure.DICT_OF_DICTS
        assert "users" in data
        assert "settings" in data
    
    def test_parse_valid_list(self):
        """Test parsing valid list JSON."""
        json_string = '''
        [
            {"id": 1, "name": "Item 1"},
            {"id": 2, "name": "Item 2"},
            {"id": 3, "name": "Item 3"}
        ]
        '''
        
        data, structure_type = self.parser.parse(json_string)
        
        assert isinstance(data, list)
        assert structure_type == DataStructure.LIST
        assert len(data) == 3
    
    def test_parse_mixed_structure(self):
        """Test parsing mixed structure JSON."""
        json_string = '''
        {
            "metadata": {"version": "1.0"},
            "data": [1, 2, 3],
            "config": "simple_value"
        }
        '''
        
        data, structure_type = self.parser.parse(json_string)
        
        assert isinstance(data, dict)
        assert structure_type == DataStructure.MIXED
    
    def test_parse_invalid_json_syntax(self):
        """Test parsing invalid JSON syntax."""
        json_string = '{"users": {"user1": {"name": "Alice"}'  # Missing closing braces
        
        with pytest.raises(ValueError, match="Invalid JSON input"):
            self.parser.parse(json_string)
    
    def test_parse_empty_json(self):
        """Test parsing empty JSON string."""
        with pytest.raises(ValueError, match="Invalid JSON input"):
            self.parser.parse("")
    
    def test_parse_primitive_root(self):
        """Test parsing primitive root type (should fail)."""
        json_string = '"just a string"'
        
        with pytest.raises(ValueError, match="Unsupported root data type"):
            self.parser.parse(json_string)
    
    def test_detect_dict_of_dicts_structure(self):
        """Test detection of dict-of-dicts structure."""
        data = {
            "section1": {"key1": "value1", "key2": "value2"},
            "section2": {"key3": "value3", "key4": "value4"},
            "section3": {"key5": "value5"}
        }
        
        structure_type = self.parser.detect_data_structure(data)
        assert structure_type == DataStructure.DICT_OF_DICTS
    
    def test_detect_mixed_dict_structure(self):
        """Test detection of mixed dictionary structure."""
        data = {
            "dict_section": {"key1": "value1"},
            "list_section": [1, 2, 3],
            "primitive_section": "simple_value"
        }
        
        structure_type = self.parser.detect_data_structure(data)
        assert structure_type == DataStructure.MIXED
    
    def test_detect_list_structure(self):
        """Test detection of list structure."""
        data = [
            {"id": 1, "name": "Item 1"},
            {"id": 2, "name": "Item 2"},
            {"id": 3, "name": "Item 3"}
        ]
        
        structure_type = self.parser.detect_data_structure(data)
        assert structure_type == DataStructure.LIST
    
    def test_detect_mixed_list_structure(self):
        """Test detection of mixed list structure."""
        data = [
            {"id": 1, "name": "Item 1"},
            [1, 2, 3],
            "simple_string"
        ]
        
        structure_type = self.parser.detect_data_structure(data)
        assert structure_type == DataStructure.MIXED
    
    def test_detect_empty_dict_structure(self):
        """Test detection of empty dictionary structure."""
        data = {}
        
        structure_type = self.parser.detect_data_structure(data)
        assert structure_type == DataStructure.DICT_OF_DICTS
    
    def test_detect_empty_list_structure(self):
        """Test detection of empty list structure."""
        data = []
        
        structure_type = self.parser.detect_data_structure(data)
        assert structure_type == DataStructure.LIST
    
    def test_validate_for_processing_valid_data(self):
        """Test validation of valid data for processing."""
        data = {"users": {"user1": {"name": "Alice"}}}
        
        result = self.parser.validate_for_processing(data, 25000)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_for_processing_large_data_warning(self):
        """Test validation with large data (should generate warning)."""
        # Create large data structure
        large_data = {}
        for i in range(1000):
            large_data[f"key_{i}"] = {"value": "x" * 100}
        
        result = self.parser.validate_for_processing(large_data, 100)  # Small max_size
        
        assert result.is_valid
        assert len(result.warnings) > 0
        assert "much larger" in result.warnings[0]
    
    def test_validate_for_processing_deep_nesting_warning(self):
        """Test validation with deep nesting (should generate warning)."""
        # Create deeply nested structure
        data = {}
        current = data
        for i in range(15):  # Create 15 levels of nesting
            current[f"level_{i}"] = {}
            current = current[f"level_{i}"]
        
        result = self.parser.validate_for_processing(data, 25000)
        
        assert result.is_valid
        assert len(result.warnings) > 0
        assert "Deep nesting detected" in result.warnings[0]
    
    def test_validate_for_processing_non_serializable_data(self):
        """Test validation with non-serializable data."""
        # Create data with non-serializable object
        class NonSerializable:
            pass
        
        data = {"valid": "data", "invalid": NonSerializable()}
        
        result = self.parser.validate_for_processing(data, 25000)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].type == ErrorType.STRUCTURE
        assert "not JSON serializable" in result.errors[0].message
    
    def test_calculate_nesting_depth_simple(self):
        """Test calculation of nesting depth for simple structure."""
        data = {"level1": {"level2": {"level3": "value"}}}
        
        depth = self.parser._calculate_nesting_depth(data)
        assert depth == 3
    
    def test_calculate_nesting_depth_list(self):
        """Test calculation of nesting depth for list structure."""
        data = [[[["deep_value"]]]]
        
        depth = self.parser._calculate_nesting_depth(data)
        assert depth == 4
    
    def test_calculate_nesting_depth_mixed(self):
        """Test calculation of nesting depth for mixed structure."""
        data = {
            "dict_branch": {"level2": {"level3": "value"}},
            "list_branch": [[[["deeper_value"]]]]
        }
        
        depth = self.parser._calculate_nesting_depth(data)
        assert depth == 5  # The list branch is deeper
    
    def test_find_large_values(self):
        """Test finding large values in data structure."""
        large_string = "x" * 1000  # 1000 character string
        data = {
            "small_value": "small",
            "large_value": large_string,
            "nested": {
                "another_large": large_string
            }
        }
        
        large_values = self.parser._find_large_values(data, 500)  # 500 byte threshold
        
        assert len(large_values) >= 2  # Should find at least 2 large values
        paths = [path for path, size in large_values]
        assert ["large_value"] in paths
        assert ["nested", "another_large"] in paths
    
    def test_get_structure_statistics(self):
        """Test getting structure statistics."""
        data = {
            "users": {
                "user1": {"name": "Alice", "age": 30},
                "user2": {"name": "Bob", "age": 25}
            },
            "items": [1, 2, 3, 4, 5],
            "config": "simple_value"
        }
        
        stats = self.parser.get_structure_statistics(data)
        
        assert stats["structure_type"] == "mixed"
        assert stats["dict_count"] == 3  # root, users, user1, user2
        assert stats["list_count"] == 1  # items list
        assert stats["primitive_count"] > 0
        assert stats["total_size"] > 0
        assert stats["max_depth"] > 0
    
    def test_get_structure_statistics_simple_dict(self):
        """Test getting statistics for simple dictionary."""
        data = {"key1": "value1", "key2": "value2"}
        
        stats = self.parser.get_structure_statistics(data)
        
        assert stats["structure_type"] == "dict-of-dicts"
        assert stats["dict_count"] == 1
        assert stats["list_count"] == 0
        assert stats["primitive_count"] == 2
        assert stats["total_keys"] == 2
    
    def test_get_structure_statistics_simple_list(self):
        """Test getting statistics for simple list."""
        data = [1, 2, 3, 4, 5]
        
        stats = self.parser.get_structure_statistics(data)
        
        assert stats["structure_type"] == "list"
        assert stats["dict_count"] == 0
        assert stats["list_count"] == 1
        assert stats["primitive_count"] == 5
        assert stats["total_items"] == 5
    
    def test_count_elements_nested_structure(self):
        """Test element counting in nested structure."""
        data = {
            "level1": {
                "level2": {
                    "items": [1, 2, {"nested_item": "value"}]
                }
            }
        }
        
        stats = {"dict_count": 0, "list_count": 0, "primitive_count": 0, 
                "total_keys": 0, "total_items": 0}
        
        self.parser._count_elements(data, stats)
        
        assert stats["dict_count"] == 3  # root, level1, level2, nested_item dict
        assert stats["list_count"] == 1  # items list
        assert stats["primitive_count"] == 3  # 1, 2, "value"
        assert stats["total_keys"] == 4  # level1, level2, items, nested_item
        assert stats["total_items"] == 3  # items in the list