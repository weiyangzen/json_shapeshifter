"""Tests for file sharding engine."""

import pytest
from json_transformer.engines.file_sharding_engine import FileShardingEngine
from json_transformer.models import ShardMetadata
from json_transformer.types import DataType


class TestFileShardingEngine:
    """Tests for FileShardingEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = FileShardingEngine()
    
    def test_create_shard_valid(self):
        """Test creating a valid shard."""
        data = {"user": {"name": "Alice", "age": 30}}
        shard_id = "shard_001"
        
        metadata = ShardMetadata(
            shard_id=shard_id,
            parent_id=None,
            child_ids=[],
            data_type=DataType.DICT,
            original_path=["users"]
        )
        
        shard = self.engine.create_shard(data, shard_id, metadata)
        
        assert shard.id == shard_id
        assert shard.data == data
        assert shard.metadata == metadata
        assert shard.size > 0
    
    def test_create_shard_id_mismatch(self):
        """Test creating shard with mismatched ID."""
        data = {"test": "data"}
        shard_id = "shard_001"
        
        metadata = ShardMetadata(
            shard_id="shard_002",  # Different ID
            parent_id=None,
            child_ids=[],
            data_type=DataType.DICT,
            original_path=[]
        )
        
        with pytest.raises(ValueError, match="Shard ID mismatch"):
            self.engine.create_shard(data, shard_id, metadata)
    
    def test_calculate_size_valid_data(self):
        """Test size calculation for valid data."""
        data = {"key": "value", "number": 42}
        size = self.engine.calculate_size(data)
        
        assert size > 0
        assert isinstance(size, int)
    
    def test_calculate_size_non_serializable(self):
        """Test size calculation for non-serializable data."""
        class NonSerializable:
            pass
        
        data = {"valid": "data", "invalid": NonSerializable()}
        size = self.engine.calculate_size(data)
        
        # Should return -1 for calculation failure
        assert size == -1
    
    def test_should_create_new_shard_within_limit(self):
        """Test shard creation decision when within size limit."""
        current_size = 1000  # 1KB
        new_data_size = 500  # 0.5KB
        max_size = 25000  # 25KB
        
        should_create = self.engine.should_create_new_shard(
            current_size, new_data_size, max_size
        )
        
        assert not should_create
    
    def test_should_create_new_shard_exceeds_limit(self):
        """Test shard creation decision when exceeding size limit."""
        current_size = 20000  # 20KB
        new_data_size = 8000   # 8KB
        max_size = 25000       # 25KB
        
        should_create = self.engine.should_create_new_shard(
            current_size, new_data_size, max_size
        )
        
        assert should_create
    
    def test_generate_shard_id(self):
        """Test shard ID generation."""
        id1 = self.engine.generate_shard_id()
        id2 = self.engine.generate_shard_id()
        
        assert id1 != id2
        assert id1.startswith("shard_")
        assert id2.startswith("shard_")
    
    def test_generate_shard_id_custom_prefix(self):
        """Test shard ID generation with custom prefix."""
        custom_id = self.engine.generate_shard_id("custom")
        
        assert custom_id.startswith("custom_")
    
    def test_split_data_by_size_small_dict(self):
        """Test splitting small dictionary that fits in one chunk."""
        data = {"key1": "value1", "key2": "value2"}
        max_size = 25000  # 25KB
        
        chunks = self.engine.split_data_by_size(data, max_size)
        
        assert len(chunks) == 1
        assert chunks[0][0] == data
        assert chunks[0][1] == []
    
    def test_split_data_by_size_large_dict(self):
        """Test splitting large dictionary into multiple chunks."""
        # Create large dictionary
        data = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}
        max_size = 1000  # 1KB (small to force splitting)
        
        chunks = self.engine.split_data_by_size(data, max_size)
        
        assert len(chunks) > 1
        
        # Verify all chunks are dictionaries
        for chunk_data, chunk_path in chunks:
            assert isinstance(chunk_data, dict)
            assert chunk_path == []
    
    def test_split_data_by_size_list(self):
        """Test splitting list data."""
        data = [{"item": f"data_{i}" * 50} for i in range(20)]
        max_size = 1000  # 1KB
        
        chunks = self.engine.split_data_by_size(data, max_size)
        
        assert len(chunks) > 1
        
        # Verify all chunks are lists
        for chunk_data, chunk_path in chunks:
            assert isinstance(chunk_data, list)
            assert chunk_path == []
    
    def test_split_data_by_size_nested_structure(self):
        """Test splitting nested data structure."""
        data = {
            "section1": {f"key_{i}": f"value_{i}" * 100 for i in range(50)},
            "section2": [{"item": f"data_{i}" * 50} for i in range(30)]
        }
        max_size = 2000  # 2KB
        
        chunks = self.engine.split_data_by_size(data, max_size)
        
        assert len(chunks) > 1
        
        # Should have chunks with different paths
        paths = [chunk_path for _, chunk_path in chunks]
        assert any(len(path) > 0 for path in paths)  # Some chunks should have nested paths
    
    def test_create_shards_from_chunks(self):
        """Test creating shards from data chunks."""
        chunks = [
            ({"chunk1": "data1"}, ["section1"]),
            ({"chunk2": "data2"}, ["section2"]),
            ([1, 2, 3], ["section3"])
        ]
        
        shards = self.engine.create_shards_from_chunks(chunks)
        
        assert len(shards) == 3
        
        # Check sequential linking
        assert shards[0].metadata.next_shard == shards[1].id
        assert shards[1].metadata.previous_shard == shards[0].id
        assert shards[1].metadata.next_shard == shards[2].id
        assert shards[2].metadata.previous_shard == shards[1].id
        
        # Check data types
        assert shards[0].metadata.data_type == DataType.DICT
        assert shards[1].metadata.data_type == DataType.DICT
        assert shards[2].metadata.data_type == DataType.LIST
    
    def test_create_shards_from_chunks_with_parent(self):
        """Test creating shards with parent ID."""
        chunks = [
            ({"chunk1": "data1"}, ["section1"]),
            ({"chunk2": "data2"}, ["section2"])
        ]
        parent_id = "parent_shard_001"
        
        shards = self.engine.create_shards_from_chunks(chunks, parent_id)
        
        assert len(shards) == 2
        
        # Check parent relationships
        for shard in shards:
            assert shard.metadata.parent_id == parent_id
    
    def test_optimize_shard_distribution_within_limits(self):
        """Test optimizing shards that are within size limits."""
        # Create shards within size limits
        chunks = [
            ({"small": "data1"}, []),
            ({"small": "data2"}, [])
        ]
        shards = self.engine.create_shards_from_chunks(chunks)
        max_size = 25000  # 25KB
        
        optimized = self.engine.optimize_shard_distribution(shards, max_size)
        
        # Should return same shards since they're within limits
        assert len(optimized) == len(shards)
    
    def test_optimize_shard_distribution_oversized(self):
        """Test optimizing oversized shards."""
        # Create an oversized shard by manually setting size
        large_data = {f"key_{i}": f"value_{i}" * 1000 for i in range(100)}
        chunks = [(large_data, [])]
        shards = self.engine.create_shards_from_chunks(chunks)
        
        # Force the shard to be oversized
        shards[0].size = 50000  # 50KB (oversized)
        
        max_size = 25000  # 25KB
        optimized = self.engine.optimize_shard_distribution(shards, max_size)
        
        # Should create more shards
        assert len(optimized) > len(shards)
    
    def test_get_sharding_statistics_empty(self):
        """Test getting statistics for empty shard list."""
        stats = self.engine.get_sharding_statistics([])
        
        assert stats["total_shards"] == 0
        assert stats["total_size"] == 0
        assert stats["average_size"] == 0
    
    def test_get_sharding_statistics_with_shards(self):
        """Test getting statistics for shard list."""
        chunks = [
            ({"data1": "value1"}, []),
            ({"data2": "value2"}, []),
            ([1, 2, 3], [])
        ]
        shards = self.engine.create_shards_from_chunks(chunks)
        
        stats = self.engine.get_sharding_statistics(shards)
        
        assert stats["total_shards"] == 3
        assert stats["total_size"] > 0
        assert stats["average_size"] > 0
        assert "size_distribution" in stats
        assert "data_type_distribution" in stats
        assert "efficiency" in stats
        
        # Check data type distribution
        assert "dict" in stats["data_type_distribution"]
        assert "list" in stats["data_type_distribution"]
    
    def test_split_dict_by_size_single_oversized_value(self):
        """Test splitting dictionary with single oversized value."""
        # Create dict with one very large value
        large_value = {f"nested_{i}": f"data_{i}" * 100 for i in range(100)}
        data = {
            "small_key": "small_value",
            "large_key": large_value
        }
        max_size = 1000  # 1KB
        
        chunks = self.engine._split_dict_by_size(data, max_size, [])
        
        assert len(chunks) > 1
        
        # Should have separate chunks for small and large values
        small_chunks = [c for c in chunks if "small_key" in str(c[0])]
        large_chunks = [c for c in chunks if c[0] != data and "small_key" not in str(c[0])]
        
        assert len(small_chunks) >= 1
        assert len(large_chunks) >= 1
    
    def test_split_list_by_size_oversized_items(self):
        """Test splitting list with oversized items."""
        # Create list with some oversized items
        large_item = {f"key_{i}": f"value_{i}" * 100 for i in range(50)}
        data = [
            {"small": "item1"},
            large_item,
            {"small": "item2"}
        ]
        max_size = 1000  # 1KB
        
        chunks = self.engine._split_list_by_size(data, max_size, [])
        
        assert len(chunks) > 1
    
    def test_calculate_efficiency(self):
        """Test efficiency calculation."""
        # Create shards with known sizes
        chunks = [
            ({"data": "small"}, []),
            ({"data": "medium" * 100}, []),
            ({"data": "large" * 500}, [])
        ]
        shards = self.engine.create_shards_from_chunks(chunks)
        
        efficiency = self.engine._calculate_efficiency(shards, 25000)
        
        assert 0.0 <= efficiency <= 1.0
    
    def test_split_data_primitive_oversized(self):
        """Test splitting primitive data that exceeds size limit."""
        # This is an edge case - primitive data that's too large
        large_string = "x" * 50000  # 50KB string
        max_size = 25000  # 25KB
        
        chunks = self.engine.split_data_by_size(large_string, max_size, ["large_string"])
        
        # Should still return the data with a warning
        assert len(chunks) == 1
        assert chunks[0][0] == large_string
        assert chunks[0][1] == ["large_string"]
    
    def test_sequential_shard_id_generation(self):
        """Test that shard IDs are generated sequentially."""
        ids = [self.engine.generate_shard_id() for _ in range(5)]
        
        # Extract numbers from IDs
        numbers = [int(id.split('_')[1]) for id in ids]
        
        # Should be sequential
        for i in range(1, len(numbers)):
            assert numbers[i] == numbers[i-1] + 1