"""Tests for data type detector."""

import pytest
from json_transformer.data_type_detector import DataTypeDetector
from json_transformer.types import DataStructure, DataType


class TestDataTypeDetector:
    """Tests for DataTypeDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DataTypeDetector()
    
    def test_detect_structure_type_dict_of_dicts(self):
        """Test detection of dict-of-dicts structure."""
        data = {
            "users": {
                "user1": {"name": "Alice", "age": 30},
                "user2": {"name": "Bob", "age": 25}
            },
            "settings": {
                "theme": "dark",
                "notifications": True
            }
        }
        
        structure_type = self.detector.detect_structure_type(data)
        assert structure_type == DataStructure.DICT_OF_DICTS
    
    def test_detect_structure_type_list(self):
        """Test detection of list structure."""
        data = [
            {"id": 1, "name": "Item 1"},
            {"id": 2, "name": "Item 2"},
            {"id": 3, "name": "Item 3"}
        ]
        
        structure_type = self.detector.detect_structure_type(data)
        assert structure_type == DataStructure.LIST
    
    def test_detect_structure_type_mixed_dict(self):
        """Test detection of mixed dictionary structure."""
        data = {
            "metadata": {"version": "1.0", "created": "2024-01-01"},
            "data": [1, 2, 3, 4, 5],
            "config": "simple_value",
            "flags": True
        }
        
        structure_type = self.detector.detect_structure_type(data)
        assert structure_type == DataStructure.MIXED
    
    def test_detect_structure_type_mixed_list(self):
        """Test detection of mixed list structure."""
        data = [
            {"id": 1, "name": "Item 1"},
            [1, 2, 3],
            "simple_string",
            42
        ]
        
        structure_type = self.detector.detect_structure_type(data)
        assert structure_type == DataStructure.MIXED
    
    def test_detect_element_type_dict(self):
        """Test detection of dictionary element type."""
        element = {"key": "value"}
        element_type = self.detector.detect_element_type(element)
        assert element_type == DataType.DICT
    
    def test_detect_element_type_list(self):
        """Test detection of list element type."""
        element = [1, 2, 3]
        element_type = self.detector.detect_element_type(element)
        assert element_type == DataType.LIST
    
    def test_detect_element_type_primitive(self):
        """Test detection of primitive element types."""
        assert self.detector.detect_element_type("string") == DataType.PRIMITIVE
        assert self.detector.detect_element_type(42) == DataType.PRIMITIVE
        assert self.detector.detect_element_type(True) == DataType.PRIMITIVE
        assert self.detector.detect_element_type(None) == DataType.PRIMITIVE
    
    def test_analyze_dict_patterns_homogeneous(self):
        """Test analysis of homogeneous dictionary patterns."""
        data = {
            "section1": {"key1": "value1", "key2": "value2"},
            "section2": {"key3": "value3", "key4": "value4"},
            "section3": {"key5": "value5", "key6": "value6"}
        }
        
        patterns = self.detector.analyze_dict_patterns(data)
        
        assert patterns["structure_type"] == DataStructure.DICT_OF_DICTS
        assert patterns["homogeneous"] is True
        assert patterns["dict_ratio"] == 1.0
        assert patterns["list_ratio"] == 0.0
        assert patterns["primitive_ratio"] == 0.0
        assert patterns["total_keys"] == 3
    
    def test_analyze_dict_patterns_mixed(self):
        """Test analysis of mixed dictionary patterns."""
        data = {
            "dict_section": {"key1": "value1"},
            "list_section": [1, 2, 3],
            "primitive_section": "simple_value"
        }
        
        patterns = self.detector.analyze_dict_patterns(data)
        
        assert patterns["structure_type"] == DataStructure.MIXED
        assert patterns["homogeneous"] is False
        assert patterns["dict_ratio"] == 1/3
        assert patterns["list_ratio"] == 1/3
        assert patterns["primitive_ratio"] == 1/3
    
    def test_analyze_dict_patterns_empty(self):
        """Test analysis of empty dictionary."""
        data = {}
        
        patterns = self.detector.analyze_dict_patterns(data)
        
        assert patterns["is_empty"] is True
        assert patterns["structure_type"] == DataStructure.DICT_OF_DICTS
        assert patterns["homogeneous"] is True
    
    def test_analyze_list_patterns_homogeneous(self):
        """Test analysis of homogeneous list patterns."""
        data = [
            {"id": 1, "name": "Item 1"},
            {"id": 2, "name": "Item 2"},
            {"id": 3, "name": "Item 3"}
        ]
        
        patterns = self.detector.analyze_list_patterns(data)
        
        assert patterns["structure_type"] == DataStructure.LIST
        assert patterns["homogeneous"] is True
        assert patterns["dict_ratio"] == 1.0
        assert patterns["list_ratio"] == 0.0
        assert patterns["primitive_ratio"] == 0.0
        assert patterns["total_items"] == 3
    
    def test_analyze_list_patterns_mixed(self):
        """Test analysis of mixed list patterns."""
        data = [
            {"id": 1, "name": "Item 1"},
            [1, 2, 3],
            "simple_string"
        ]
        
        patterns = self.detector.analyze_list_patterns(data)
        
        assert patterns["structure_type"] == DataStructure.MIXED
        assert patterns["homogeneous"] is False
        assert patterns["dict_ratio"] == 1/3
        assert patterns["list_ratio"] == 1/3
        assert patterns["primitive_ratio"] == 1/3
    
    def test_analyze_list_patterns_empty(self):
        """Test analysis of empty list."""
        data = []
        
        patterns = self.detector.analyze_list_patterns(data)
        
        assert patterns["is_empty"] is True
        assert patterns["structure_type"] == DataStructure.LIST
        assert patterns["homogeneous"] is True
    
    def test_detect_nested_structures(self):
        """Test detection of nested structures."""
        data = {
            "level1": {
                "level2": {
                    "items": [1, 2, {"nested": "value"}]
                }
            },
            "simple": "value"
        }
        
        nested_structures = self.detector.detect_nested_structures(data)
        
        # Should find: root dict, level1 dict, level2 dict, items list, nested dict
        assert len(nested_structures) >= 4
        
        # Check that we found structures at different depths
        depths = [s["depth"] for s in nested_structures]
        assert 0 in depths  # Root level
        assert 1 in depths  # level1
        assert 2 in depths  # level2
        assert 3 in depths  # items list
    
    def test_identify_sharding_candidates(self):
        """Test identification of sharding candidates."""
        # Create data with some large structures
        large_dict = {f"key_{i}": f"value_{i}" * 100 for i in range(50)}
        data = {
            "small_section": {"key": "value"},
            "large_section": large_dict,
            "medium_section": {f"key_{i}": f"value_{i}" for i in range(10)}
        }
        
        candidates = self.detector.identify_sharding_candidates(data, 1000)  # 1KB threshold
        
        # Should identify large_section as a candidate
        assert len(candidates) > 0
        
        # Check that candidates are sorted by priority
        if len(candidates) > 1:
            assert candidates[0]["sharding_priority"] >= candidates[1]["sharding_priority"]
    
    def test_get_processing_recommendations_simple(self):
        """Test processing recommendations for simple structure."""
        data = {
            "users": {
                "user1": {"name": "Alice"},
                "user2": {"name": "Bob"}
            }
        }
        
        recommendations = self.detector.get_processing_recommendations(data, 25000)
        
        assert recommendations["structure_type"] == "dict-of-dicts"
        assert recommendations["recommended_processor"] == "DictProcessor"
        assert recommendations["complexity_score"] >= 0
        assert "processing_strategy" in recommendations
    
    def test_get_processing_recommendations_complex(self):
        """Test processing recommendations for complex structure."""
        # Create complex nested structure
        data = {
            "metadata": {"version": "1.0", "created": "2024-01-01"},
            "users": {f"user_{i}": {"name": f"User {i}", "data": list(range(100))} for i in range(50)},
            "config": [{"setting": f"value_{i}"} for i in range(100)],
            "simple": "value"
        }
        
        recommendations = self.detector.get_processing_recommendations(data, 1000)  # Small threshold
        
        assert recommendations["structure_type"] == "mixed"
        assert recommendations["recommended_processor"] == "MixedProcessor"
        assert recommendations["sharding_needed"] is True
        assert recommendations["estimated_shards"] > 1
        assert recommendations["complexity_score"] > 1.0
    
    def test_calculate_element_depth(self):
        """Test calculation of element depth."""
        # Simple element
        assert self.detector._calculate_element_depth("simple") == 0
        
        # Nested dict
        nested_dict = {"level1": {"level2": {"level3": "value"}}}
        assert self.detector._calculate_element_depth(nested_dict) == 3
        
        # Nested list
        nested_list = [[[["deep_value"]]]]
        assert self.detector._calculate_element_depth(nested_list) == 4
    
    def test_get_data_at_path(self):
        """Test getting data at specific path."""
        data = {
            "level1": {
                "level2": {
                    "items": [1, 2, {"nested": "value"}]
                }
            }
        }
        
        # Test dict path
        result = self.detector._get_data_at_path(data, ["level1", "level2"])
        assert result == {"items": [1, 2, {"nested": "value"}]}
        
        # Test list path
        result = self.detector._get_data_at_path(data, ["level1", "level2", "items", "[2]"])
        assert result == {"nested": "value"}
        
        # Test invalid path
        result = self.detector._get_data_at_path(data, ["nonexistent"])
        assert result is None
    
    def test_calculate_sharding_priority(self):
        """Test calculation of sharding priority."""
        structure = {
            "path": ["test"],
            "type": DataType.DICT,
            "size": 100,
            "depth": 1
        }
        
        # Large size should give high priority
        priority_large = self.detector._calculate_sharding_priority(structure, 10000, 1000)
        priority_small = self.detector._calculate_sharding_priority(structure, 1500, 1000)
        
        assert priority_large > priority_small
        
        # Dict should have higher priority than list
        dict_structure = structure.copy()
        dict_structure["type"] = DataType.DICT
        
        list_structure = structure.copy()
        list_structure["type"] = DataType.LIST
        
        dict_priority = self.detector._calculate_sharding_priority(dict_structure, 5000, 1000)
        list_priority = self.detector._calculate_sharding_priority(list_structure, 5000, 1000)
        
        assert dict_priority > list_priority
    
    def test_estimate_shard_count(self):
        """Test estimation of shard count."""
        # Small data
        small_data = {"key": "value"}
        count = self.detector._estimate_shard_count(small_data, 25000)
        assert count == 1
        
        # Large data (create large structure)
        large_data = {f"key_{i}": f"value_{i}" * 1000 for i in range(100)}
        count = self.detector._estimate_shard_count(large_data, 1000)
        assert count > 1
    
    def test_calculate_complexity_score(self):
        """Test calculation of complexity score."""
        # Simple patterns
        simple_patterns = {
            "nesting_levels": {"max": 1, "min": 1, "avg": 1},
            "homogeneous": True,
            "structure_type": DataStructure.DICT_OF_DICTS,
            "total_keys": 5
        }
        
        simple_score = self.detector._calculate_complexity_score(simple_patterns)
        
        # Complex patterns
        complex_patterns = {
            "nesting_levels": {"max": 5, "min": 1, "avg": 3},
            "homogeneous": False,
            "structure_type": DataStructure.MIXED,
            "total_keys": 1000
        }
        
        complex_score = self.detector._calculate_complexity_score(complex_patterns)
        
        assert complex_score > simple_score
    
    def test_recommend_processor(self):
        """Test processor recommendations."""
        assert self.detector._recommend_processor(DataStructure.DICT_OF_DICTS) == "DictProcessor"
        assert self.detector._recommend_processor(DataStructure.LIST) == "ListProcessor"
        assert self.detector._recommend_processor(DataStructure.MIXED) == "MixedProcessor"
    
    def test_recommend_processing_strategy(self):
        """Test processing strategy recommendations."""
        # Simple case - no sharding needed
        simple_patterns = {"nesting_levels": {"max": 1}}
        strategy = self.detector._recommend_processing_strategy(
            DataStructure.DICT_OF_DICTS, simple_patterns, []
        )
        assert strategy == "simple_processing"
        
        # Complex case - many candidates
        complex_patterns = {"nesting_levels": {"max": 5}}
        many_candidates = [{"path": [f"path_{i}"]} for i in range(15)]
        strategy = self.detector._recommend_processing_strategy(
            DataStructure.MIXED, complex_patterns, many_candidates
        )
        assert strategy == "aggressive_sharding"