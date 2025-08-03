"""Tests for list processor."""

import pytest
from json_transformer.processors.list_processor import ListProcessor
from json_transformer.types import DataStructure, DataType


class TestListProcessor:
    """Tests for ListProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ListProcessor()
    
    def test_process_simple_list(self):
        """Test processing simple list."""
        data = [
            {"id": 1, "name": "Item 1"},
            {"id": 2, "name": "Item 2"},
            {"id": 3, "name": "Item 3"}
        ]
        
        result = self.processor.process(data, 25000)
        
        assert result.shards
        assert result.root_shard_id
        assert result.metadata.data_structure == DataStructure.LIST
        assert result.metadata.shard_count > 0
    
    def test_process_non_list_data(self):
        """Test processing non-list data (should fail)."""
        data = {"key": "value"}
        
        with pytest.raises(ValueError, match="ListProcessor expects list data"):
            self.processor.process(data, 25000)
    
    def test_process_empty_list(self):
        """Test processing empty list."""
        data = []
        
        result = self.processor.process(data, 25000)
        
        assert len(result.shards) == 1  # Should create one shard for empty list
        assert result.shards[0].data == []
        assert result.shards[0].metadata.data_type == DataType.LIST
    
    def test_process_large_list(self):
        """Test processing large list that needs splitting."""
        # Create large list
        data = [{"item": f"data_{i}" * 100} for i in range(100)]
        
        result = self.processor.process(data, 2000)  # Small max size to force splitting
        
        assert len(result.shards) > 1
        
        # Verify sequential linking
        for i in range(len(result.shards) - 1):
            assert result.shards[i].metadata.next_shard == result.shards[i + 1].id
            assert result.shards[i + 1].metadata.previous_shard == result.shards[i].id
    
    def test_process_list_with_nested_dicts(self):
        """Test processing list containing nested dictionaries."""
        data = [
            {
                "user": {"name": "Alice", "profile": {"age": 30, "city": "NYC"}},
                "settings": {"theme": "dark"}
            },
            {
                "user": {"name": "Bob", "profile": {"age": 25, "city": "SF"}},
                "settings": {"theme": "light"}
            }
        ]
        
        result = self.processor.process(data, 25000)
        
        assert len(result.shards) >= 1
        
        # Should handle nested structures appropriately
        for shard in result.shards:
            assert shard.metadata.data_type in [DataType.LIST, DataType.DICT]
    
    def test_process_list_with_nested_lists(self):
        """Test processing list containing nested lists."""
        data = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        result = self.processor.process(data, 25000)
        
        assert len(result.shards) >= 1
        assert result.metadata.data_structure == DataStructure.LIST
    
    def test_analyze_list_structure_simple(self):
        """Test analysis of simple list structure."""
        data = [1, 2, 3, 4, 5]
        
        analysis = self.processor._analyze_list_structure(data)
        
        assert not analysis["has_nested_structures"]
        assert analysis["homogeneous"]
        assert analysis["total_items"] == 5
        assert "int" in analysis["item_types"]
    
    def test_analyze_list_structure_nested(self):
        """Test analysis of nested list structure."""
        data = [
            {"key": "value"},
            [1, 2, 3],
            "simple_string"
        ]
        
        analysis = self.processor._analyze_list_structure(data)
        
        assert analysis["has_nested_structures"]
        assert not analysis["homogeneous"]
        assert analysis["total_items"] == 3
        assert len(analysis["item_types"]) == 3  # dict, list, str
    
    def test_analyze_list_structure_empty(self):
        """Test analysis of empty list."""
        data = []
        
        analysis = self.processor._analyze_list_structure(data)
        
        assert not analysis["has_nested_structures"]
        assert analysis["homogeneous"]
        assert analysis["total_items"] == 0
        assert analysis["max_item_size"] == 0
    
    def test_process_simple_list_method(self):
        """Test processing simple list without nested structures."""
        data = ["item1", "item2", "item3"]
        
        shards = self.processor._process_simple_list(data, 25000)
        
        assert len(shards) >= 1
        
        # Check that all shards are list type
        for shard in shards:
            assert shard.metadata.data_type == DataType.LIST
            assert isinstance(shard.data, list)
    
    def test_split_list_by_size_small_list(self):
        """Test splitting small list that fits in one chunk."""
        data = [1, 2, 3, 4, 5]
        
        chunks = self.processor._split_list_by_size(data, 25000)
        
        assert len(chunks) == 1
        assert chunks[0] == data
    
    def test_split_list_by_size_large_list(self):
        """Test splitting large list into multiple chunks."""
        # Create list with large items
        data = [{"large_item": "x" * 500} for _ in range(20)]
        
        chunks = self.processor._split_list_by_size(data, 2000)  # Small max size
        
        assert len(chunks) > 1
        
        # Verify all items are preserved
        total_items = sum(len(chunk) for chunk in chunks)
        assert total_items == len(data)
    
    def test_create_chunk_shards_single_shard(self):
        """Test creating shards from chunk that fits in one shard."""
        chunk = [{"item": 1}, {"item": 2}]
        start_index = 0
        
        shards = self.processor._create_chunk_shards(chunk, start_index, 25000)
        
        assert len(shards) == 1
        assert shards[0].data == chunk
        assert shards[0].metadata.data_type == DataType.LIST
    
    def test_create_chunk_shards_multiple_shards(self):
        """Test creating shards from chunk that needs splitting."""
        # Create large chunk
        chunk = [{"large_item": "x" * 1000} for _ in range(10)]
        start_index = 0
        
        shards = self.processor._create_chunk_shards(chunk, start_index, 2000)  # Small max size
        
        assert len(shards) > 1
        
        # Verify all items are preserved
        total_items = sum(len(shard.data) for shard in shards)
        assert total_items == len(chunk)
    
    def test_handle_oversized_dict_item(self):
        """Test handling oversized dictionary item."""
        large_dict = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}
        index = 5
        
        shards = self.processor._handle_oversized_item(large_dict, index, 1000)
        
        assert len(shards) >= 1
        
        # Check that original path includes list index
        for shard in shards:
            assert f"[{index}]" in shard.metadata.original_path[0]
    
    def test_handle_oversized_list_item(self):
        """Test handling oversized list item."""
        large_list = [{"item": f"data_{i}" * 50} for i in range(50)]
        index = 3
        
        shards = self.processor._handle_oversized_item(large_list, index, 1000)
        
        assert len(shards) >= 1
        
        # Check that original path includes list index
        for shard in shards:
            assert f"[{index}]" in shard.metadata.original_path[0]
    
    def test_handle_oversized_primitive_item(self):
        """Test handling oversized primitive item."""
        large_string = "x" * 50000  # 50KB string
        index = 2
        
        shards = self.processor._handle_oversized_item(large_string, index, 25000)
        
        assert len(shards) == 1
        assert shards[0].metadata.data_type == DataType.PRIMITIVE
        assert "_warning" in shards[0].data
        assert shards[0].data["_original_index"] == index
        assert shards[0].data["_item"] == large_string
    
    def test_setup_sequential_linking(self):
        """Test setting up sequential linking between shards."""
        # Create some test shards
        data_chunks = [["item1"], ["item2"], ["item3"]]
        shards = []
        
        for chunk in data_chunks:
            shards.extend(self.processor._create_chunk_shards(chunk, 0, 25000))
        
        # Set up linking
        self.processor._setup_sequential_linking(shards)
        
        # Verify linking
        for i in range(len(shards)):
            if i > 0:
                assert shards[i].metadata.previous_shard == shards[i-1].id
            else:
                assert shards[i].metadata.previous_shard is None
            
            if i < len(shards) - 1:
                assert shards[i].metadata.next_shard == shards[i+1].id
            else:
                assert shards[i].metadata.next_shard is None
    
    def test_create_empty_list_shard(self):
        """Test creating shard for empty list."""
        shard = self.processor._create_empty_list_shard()
        
        assert shard.data == []
        assert shard.metadata.data_type == DataType.LIST
        assert shard.metadata.original_path == ["[]"]
    
    def test_optimize_list_processing_simple(self):
        """Test optimization analysis for simple list."""
        data = [1, 2, 3, 4, 5]
        
        optimization = self.processor.optimize_list_processing(data, 25000)
        
        assert "total_size" in optimization
        assert "estimated_shards" in optimization
        assert "item_count" in optimization
        assert "analysis" in optimization
        assert "optimization_suggestions" in optimization
        
        assert optimization["item_count"] == 5
    
    def test_optimize_list_processing_empty(self):
        """Test optimization analysis for empty list."""
        data = []
        
        optimization = self.processor.optimize_list_processing(data, 25000)
        
        assert optimization["total_size"] == 0
        assert optimization["estimated_shards"] == 1
        assert optimization["item_count"] == 0
        assert len(optimization["optimization_suggestions"]) == 0
    
    def test_optimize_list_processing_large_items(self):
        """Test optimization analysis with large items."""
        # Create list with some large items
        large_item = {"data": "x" * 20000}  # 20KB item
        data = [{"small": "item"}, large_item, {"another": "small"}]
        
        optimization = self.processor.optimize_list_processing(data, 25000)
        
        assert len(optimization["large_items"]) > 0
        assert any(s["type"] == "split_large_items" for s in optimization["optimization_suggestions"])
    
    def test_optimize_list_processing_mixed_types(self):
        """Test optimization analysis with mixed item types."""
        data = [
            {"dict": "item"},
            [1, 2, 3],
            "string_item",
            42
        ]
        
        optimization = self.processor.optimize_list_processing(data, 25000)
        
        assert not optimization["analysis"]["homogeneous"]
        assert any(s["type"] == "group_similar_items" for s in optimization["optimization_suggestions"])
    
    def test_optimize_list_processing_nested_structures(self):
        """Test optimization analysis with nested structures."""
        data = [
            {"nested": {"deep": {"structure": "value"}}},
            [[[["deeply_nested"]]]],
            "simple"
        ]
        
        optimization = self.processor.optimize_list_processing(data, 25000)
        
        assert optimization["analysis"]["has_nested_structures"]
        assert any(s["type"] == "flatten_nested_structures" for s in optimization["optimization_suggestions"])
    
    def test_optimize_list_processing_very_large_list(self):
        """Test optimization analysis with very large list."""
        data = [f"item_{i}" for i in range(15000)]
        
        optimization = self.processor.optimize_list_processing(data, 25000)
        
        assert optimization["item_count"] == 15000
        assert any(s["type"] == "batch_processing" for s in optimization["optimization_suggestions"])
    
    def test_validate_list_ordering_valid(self):
        """Test validation of valid list ordering."""
        data = ["item1", "item2", "item3"]
        result = self.processor.process(data, 25000)
        
        is_valid = self.processor.validate_list_ordering(result.shards)
        
        assert is_valid
    
    def test_validate_list_ordering_empty(self):
        """Test validation of empty shard list."""
        is_valid = self.processor.validate_list_ordering([])
        
        assert is_valid
    
    def test_validate_list_ordering_broken(self):
        """Test validation of broken list ordering."""
        data = ["item1", "item2", "item3"]
        result = self.processor.process(data, 25000)
        
        # Break the linking
        if len(result.shards) > 1:
            result.shards[1].metadata.previous_shard = "wrong_id"
            
            is_valid = self.processor.validate_list_ordering(result.shards)
            assert not is_valid
    
    def test_get_list_statistics_empty(self):
        """Test getting statistics for empty shard list."""
        stats = self.processor.get_list_statistics([])
        
        assert stats["total_shards"] == 0
        assert stats["total_items"] == 0
        assert stats["average_items_per_shard"] == 0
        assert stats["sequential_integrity"]
    
    def test_get_list_statistics_with_shards(self):
        """Test getting statistics for shard list."""
        data = [f"item_{i}" for i in range(10)]
        result = self.processor.process(data, 25000)
        
        stats = self.processor.get_list_statistics(result.shards)
        
        assert stats["total_shards"] == len(result.shards)
        assert stats["total_items"] == 10
        assert stats["average_items_per_shard"] > 0
        assert stats["sequential_integrity"]
        assert "size_distribution" in stats
    
    def test_process_list_with_non_serializable_items(self):
        """Test processing list with non-serializable items."""
        class NonSerializable:
            pass
        
        data = [
            {"valid": "item1"},
            NonSerializable(),  # This should be skipped
            {"valid": "item2"}
        ]
        
        # Should not crash, but may skip non-serializable items
        result = self.processor.process(data, 25000)
        
        assert len(result.shards) >= 1
    
    def test_process_list_with_mixed_nested_structures(self):
        """Test processing list with various nested structure types."""
        data = [
            {"simple": "dict"},
            {"nested": {"dict": {"deep": "value"}}},
            [1, 2, 3],
            [[4, 5], [6, 7]],
            "simple_string",
            42
        ]
        
        result = self.processor.process(data, 25000)
        
        assert len(result.shards) >= 1
        assert result.metadata.data_structure == DataStructure.LIST