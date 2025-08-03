"""Tests for size calculator utilities."""

import pytest
import json
from json_transformer.utils.size_calculator import SizeCalculator
from json_transformer.types import DataType


class TestSizeCalculator:
    """Tests for SizeCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = SizeCalculator()
    
    def test_calculate_json_size_simple(self):
        """Test JSON size calculation for simple data."""
        data = {"key": "value", "number": 42}
        size = self.calculator.calculate_json_size(data)
        
        # Verify by comparing with actual JSON serialization
        expected_size = len(json.dumps(data, separators=(',', ':')).encode('utf-8'))
        assert size == expected_size
    
    def test_calculate_json_size_unicode(self):
        """Test JSON size calculation with Unicode characters."""
        data = {"message": "Hello 世界", "emoji": "🌍"}
        size = self.calculator.calculate_json_size(data)
        
        # Unicode characters should be properly counted in UTF-8
        expected_size = len(json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
        assert size == expected_size
    
    def test_calculate_json_size_non_serializable(self):
        """Test JSON size calculation with non-serializable data."""
        class NonSerializable:
            pass
        
        data = {"valid": "data", "invalid": NonSerializable()}
        
        with pytest.raises(ValueError, match="not JSON serializable"):
            self.calculator.calculate_json_size(data)
    
    def test_calculate_formatted_json_size(self):
        """Test formatted JSON size calculation."""
        data = {"key": "value", "nested": {"inner": "data"}}
        
        compact_size = self.calculator.calculate_json_size(data)
        formatted_size = self.calculator.calculate_formatted_json_size(data, indent=2)
        
        # Formatted JSON should be larger due to whitespace
        assert formatted_size > compact_size
    
    def test_calculate_metadata_overhead(self):
        """Test metadata overhead calculation."""
        overhead = self.calculator.calculate_metadata_overhead(
            shard_id="shard_001",
            parent_id="shard_000",
            child_ids=["shard_002", "shard_003"],
            data_type=DataType.DICT,
            original_path=["users", "profiles"]
        )
        
        # Should be a reasonable size for metadata
        assert overhead > 50  # At least 50 bytes
        assert overhead < 500  # Less than 500 bytes for simple metadata
    
    def test_calculate_file_schema_size(self):
        """Test file schema size calculation."""
        data = {"user": {"name": "Alice", "age": 30}}
        metadata_size = 150  # Assume 150 bytes of metadata
        
        schema_size = self.calculator.calculate_file_schema_size(data, metadata_size)
        data_size = self.calculator.calculate_json_size(data)
        
        # Schema size should be data + metadata + schema overhead
        assert schema_size > data_size + metadata_size
    
    def test_calculate_total_file_size(self):
        """Test total file size calculation with all overhead."""
        data = {"user": {"name": "Alice", "age": 30}}
        metadata_size = 150
        
        total_size = self.calculator.calculate_total_file_size(data, metadata_size, formatted=True)
        schema_size = self.calculator.calculate_file_schema_size(data, metadata_size)
        
        # Total size should include buffer and filesystem overhead
        assert total_size > schema_size
        assert total_size >= schema_size + self.calculator.FILE_SYSTEM_OVERHEAD
    
    def test_estimate_shard_count_small_data(self):
        """Test shard count estimation for small data."""
        total_size = 1000  # 1KB
        max_shard_size = 25000  # 25KB
        
        shard_count = self.calculator.estimate_shard_count(total_size, max_shard_size)
        
        assert shard_count == 1  # Should fit in one shard
    
    def test_estimate_shard_count_large_data(self):
        """Test shard count estimation for large data."""
        total_size = 100000  # 100KB
        max_shard_size = 25000  # 25KB
        
        shard_count = self.calculator.estimate_shard_count(total_size, max_shard_size)
        
        assert shard_count > 1  # Should need multiple shards
        assert shard_count <= 6  # Should be reasonable given overhead
    
    def test_calculate_size_breakdown(self):
        """Test detailed size breakdown calculation."""
        data = {"users": {"user1": {"name": "Alice", "age": 30}}}
        
        breakdown = self.calculator.calculate_size_breakdown(
            data=data,
            shard_id="shard_001",
            parent_id=None,
            child_ids=["shard_002"],
            data_type=DataType.DICT,
            original_path=["users"]
        )
        
        # Check that all expected components are present
        expected_keys = [
            "data_size", "metadata_size", "schema_overhead", 
            "schema_size", "formatted_size", "formatting_buffer",
            "filesystem_overhead", "total_size"
        ]
        
        for key in expected_keys:
            assert key in breakdown
            assert breakdown[key] >= 0
        
        # Check logical relationships
        assert breakdown["total_size"] >= breakdown["formatted_size"]
        assert breakdown["formatted_size"] >= breakdown["schema_size"]
        assert breakdown["schema_size"] >= breakdown["data_size"]
    
    def test_will_exceed_size_limit_within_limit(self):
        """Test size limit check when within limit."""
        current_data = {"existing": "data"}
        new_data = {"new": "data"}
        max_size = 25000  # 25KB
        
        will_exceed = self.calculator.will_exceed_size_limit(
            current_data, new_data, max_size
        )
        
        assert not will_exceed
    
    def test_will_exceed_size_limit_exceeds_limit(self):
        """Test size limit check when exceeding limit."""
        # Create large data structures
        current_data = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}
        new_data = {f"new_key_{i}": f"new_value_{i}" * 100 for i in range(100)}
        max_size = 1000  # 1KB (small limit)
        
        will_exceed = self.calculator.will_exceed_size_limit(
            current_data, new_data, max_size
        )
        
        assert will_exceed
    
    def test_will_exceed_size_limit_non_serializable(self):
        """Test size limit check with non-serializable data."""
        class NonSerializable:
            pass
        
        current_data = {"valid": "data"}
        new_data = {"invalid": NonSerializable()}
        max_size = 25000
        
        # Should return True (assume exceeds limit) for non-serializable data
        will_exceed = self.calculator.will_exceed_size_limit(
            current_data, new_data, max_size
        )
        
        assert will_exceed
    
    def test_calculate_memory_usage(self):
        """Test memory usage calculation."""
        data = {
            "users": {
                "user1": {"name": "Alice", "age": 30},
                "user2": {"name": "Bob", "age": 25}
            },
            "items": [1, 2, 3, 4, 5]
        }
        
        memory_info = self.calculator.calculate_memory_usage(data)
        
        expected_keys = ["shallow_memory", "deep_memory", "json_size", "memory_to_json_ratio"]
        for key in expected_keys:
            assert key in memory_info
        
        # Deep memory should be >= shallow memory
        assert memory_info["deep_memory"] >= memory_info["shallow_memory"]
        
        # JSON size should be positive for serializable data
        assert memory_info["json_size"] > 0
        
        # Memory to JSON ratio should be reasonable
        assert memory_info["memory_to_json_ratio"] > 0
    
    def test_calculate_deep_memory_usage_circular_reference(self):
        """Test deep memory calculation with circular references."""
        data = {"a": {}, "b": {}}
        data["a"]["ref"] = data["b"]
        data["b"]["ref"] = data["a"]
        
        # Should not crash due to circular reference
        memory_usage = self.calculator._calculate_deep_memory_usage(data)
        
        assert memory_usage > 0
    
    def test_optimize_for_size_limit_no_optimization_needed(self):
        """Test optimization when no optimization is needed."""
        data = {"small": "data"}
        max_size = 25000  # 25KB
        
        optimization = self.calculator.optimize_for_size_limit(data, max_size)
        
        assert not optimization["needs_optimization"]
        assert optimization["current_size"] <= max_size
        assert len(optimization["suggestions"]) == 0
    
    def test_optimize_for_size_limit_optimization_needed(self):
        """Test optimization when optimization is needed."""
        # Create large data structure
        large_data = {f"key_{i}": f"value_{i}" * 1000 for i in range(100)}
        max_size = 1000  # 1KB (small limit)
        
        optimization = self.calculator.optimize_for_size_limit(large_data, max_size)
        
        assert optimization["needs_optimization"]
        assert optimization["current_size"] > max_size
        assert optimization["size_reduction_needed"] > 0
        assert len(optimization["suggestions"]) > 0
    
    def test_optimize_for_size_limit_list_data(self):
        """Test optimization suggestions for list data."""
        # Create large list
        large_list = [{"item": f"data_{i}" * 100} for i in range(50)]
        max_size = 1000  # 1KB
        
        optimization = self.calculator.optimize_for_size_limit(large_list, max_size)
        
        assert optimization["needs_optimization"]
        
        # Should suggest splitting the list
        suggestions = optimization["suggestions"]
        assert any(s["type"] == "split_list" for s in suggestions)
    
    def test_calculate_metadata_overhead_different_types(self):
        """Test metadata overhead for different data types."""
        dict_overhead = self.calculator.calculate_metadata_overhead(
            "shard_001", None, [], DataType.DICT, []
        )
        
        list_overhead = self.calculator.calculate_metadata_overhead(
            "shard_001", None, [], DataType.LIST, []
        )
        
        primitive_overhead = self.calculator.calculate_metadata_overhead(
            "shard_001", None, [], DataType.PRIMITIVE, []
        )
        
        # All should be similar sizes (only dataType field differs)
        assert abs(dict_overhead - list_overhead) < 10
        assert abs(dict_overhead - primitive_overhead) < 10
    
    def test_calculate_metadata_overhead_complex_structure(self):
        """Test metadata overhead with complex structure."""
        simple_overhead = self.calculator.calculate_metadata_overhead(
            "shard_001", None, [], DataType.DICT, []
        )
        
        complex_overhead = self.calculator.calculate_metadata_overhead(
            "shard_001", 
            "parent_shard_000",
            ["child_001", "child_002", "child_003"],
            DataType.DICT,
            ["very", "deep", "nested", "path", "structure"]
        )
        
        # Complex metadata should be larger
        assert complex_overhead > simple_overhead
    
    def test_file_schema_size_consistency(self):
        """Test consistency between different size calculation methods."""
        data = {"test": {"nested": {"data": "value"}}}
        metadata_size = 200
        
        # Calculate using both methods
        schema_size = self.calculator.calculate_file_schema_size(data, metadata_size)
        formatted_size = self.calculator.calculate_file_schema_size_formatted(data, metadata_size)
        
        # Formatted should be larger
        assert formatted_size >= schema_size
        
        # Both should be reasonable
        assert schema_size > 0
        assert formatted_size > 0