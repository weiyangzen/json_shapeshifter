"""Tests for metadata manager."""

import pytest
from json_transformer.engines.metadata_manager import MetadataManager
from json_transformer.types import DataType, DataStructure


class TestMetadataManager:
    """Tests for MetadataManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = MetadataManager()
    
    def test_create_shard_metadata(self):
        """Test creating shard metadata."""
        metadata = self.manager.create_shard_metadata(
            shard_id="shard_001",
            parent_id="shard_000",
            data_type=DataType.DICT,
            original_path=["users", "profiles"],
            child_ids=["shard_002", "shard_003"]
        )
        
        assert metadata.shard_id == "shard_001"
        assert metadata.parent_id == "shard_000"
        assert metadata.data_type == DataType.DICT
        assert metadata.original_path == ["users", "profiles"]
        assert metadata.child_ids == ["shard_002", "shard_003"]
        
        # Check that it's registered
        assert self.manager.get_shard_metadata("shard_001") == metadata
    
    def test_create_global_metadata(self):
        """Test creating global metadata."""
        global_metadata = self.manager.create_global_metadata(
            version="1.0.0",
            original_size=1024000,
            shard_count=5,
            root_shard="shard_001",
            data_structure=DataStructure.DICT_OF_DICTS
        )
        
        assert global_metadata.version == "1.0.0"
        assert global_metadata.original_size == 1024000
        assert global_metadata.shard_count == 5
        assert global_metadata.root_shard == "shard_001"
        assert global_metadata.data_structure == DataStructure.DICT_OF_DICTS
        assert global_metadata.created_at is not None
    
    def test_establish_parent_child_relationship(self):
        """Test establishing parent-child relationships."""
        # Create parent and child metadata
        parent_metadata = self.manager.create_shard_metadata(
            "parent_001", None, DataType.DICT, ["root"]
        )
        child_metadata = self.manager.create_shard_metadata(
            "child_001", None, DataType.DICT, ["root", "child"]
        )
        
        # Establish relationship
        self.manager.establish_parent_child_relationship("parent_001", "child_001")
        
        # Check parent's child list
        assert "child_001" in parent_metadata.child_ids
        
        # Check child's parent
        assert child_metadata.parent_id == "parent_001"
    
    def test_establish_sequential_linking(self):
        """Test establishing sequential linking."""
        # Create multiple shards
        shard_ids = ["shard_001", "shard_002", "shard_003"]
        
        for shard_id in shard_ids:
            self.manager.create_shard_metadata(
                shard_id, None, DataType.LIST, [f"item_{shard_id}"]
            )
        
        # Establish sequential linking
        self.manager.establish_sequential_linking(shard_ids)
        
        # Check linking
        metadata_001 = self.manager.get_shard_metadata("shard_001")
        metadata_002 = self.manager.get_shard_metadata("shard_002")
        metadata_003 = self.manager.get_shard_metadata("shard_003")
        
        assert metadata_001.previous_shard is None
        assert metadata_001.next_shard == "shard_002"
        
        assert metadata_002.previous_shard == "shard_001"
        assert metadata_002.next_shard == "shard_003"
        
        assert metadata_003.previous_shard == "shard_002"
        assert metadata_003.next_shard is None
    
    def test_update_shard_metadata(self):
        """Test updating shard metadata."""
        metadata = self.manager.create_shard_metadata(
            "shard_001", None, DataType.DICT, ["root"]
        )
        
        # Update metadata
        success = self.manager.update_shard_metadata(
            "shard_001",
            parent_id="new_parent",
            child_ids=["new_child_001", "new_child_002"]
        )
        
        assert success
        assert metadata.parent_id == "new_parent"
        assert metadata.child_ids == ["new_child_001", "new_child_002"]
    
    def test_update_nonexistent_shard_metadata(self):
        """Test updating metadata for non-existent shard."""
        success = self.manager.update_shard_metadata(
            "nonexistent_shard",
            parent_id="some_parent"
        )
        
        assert not success
    
    def test_update_shard_metadata_invalid_field(self):
        """Test updating shard metadata with invalid field."""
        self.manager.create_shard_metadata(
            "shard_001", None, DataType.DICT, ["root"]
        )
        
        # Try to update invalid field (should be ignored)
        success = self.manager.update_shard_metadata(
            "shard_001",
            invalid_field="invalid_value",
            parent_id="valid_parent"  # This should work
        )
        
        assert success
        metadata = self.manager.get_shard_metadata("shard_001")
        assert metadata.parent_id == "valid_parent"
        assert not hasattr(metadata, "invalid_field")
    
    def test_get_children_metadata(self):
        """Test getting children metadata."""
        # Create parent and children
        parent_metadata = self.manager.create_shard_metadata(
            "parent_001", None, DataType.DICT, ["root"]
        )
        child1_metadata = self.manager.create_shard_metadata(
            "child_001", "parent_001", DataType.DICT, ["root", "child1"]
        )
        child2_metadata = self.manager.create_shard_metadata(
            "child_002", "parent_001", DataType.DICT, ["root", "child2"]
        )
        
        # Establish relationships
        self.manager.establish_parent_child_relationship("parent_001", "child_001")
        self.manager.establish_parent_child_relationship("parent_001", "child_002")
        
        # Get children metadata
        children = self.manager.get_children_metadata("parent_001")
        
        assert len(children) == 2
        child_ids = [child.shard_id for child in children]
        assert "child_001" in child_ids
        assert "child_002" in child_ids
    
    def test_get_children_metadata_nonexistent_parent(self):
        """Test getting children metadata for non-existent parent."""
        children = self.manager.get_children_metadata("nonexistent_parent")
        
        assert len(children) == 0
    
    def test_get_sequential_chain(self):
        """Test getting sequential chain."""
        # Create sequential shards
        shard_ids = ["shard_001", "shard_002", "shard_003", "shard_004"]
        
        for shard_id in shard_ids:
            self.manager.create_shard_metadata(
                shard_id, None, DataType.LIST, [f"item_{shard_id}"]
            )
        
        self.manager.establish_sequential_linking(shard_ids)
        
        # Get chain starting from first shard
        chain = self.manager.get_sequential_chain("shard_001")
        
        assert chain == shard_ids
    
    def test_get_sequential_chain_from_middle(self):
        """Test getting sequential chain starting from middle."""
        # Create sequential shards
        shard_ids = ["shard_001", "shard_002", "shard_003", "shard_004"]
        
        for shard_id in shard_ids:
            self.manager.create_shard_metadata(
                shard_id, None, DataType.LIST, [f"item_{shard_id}"]
            )
        
        self.manager.establish_sequential_linking(shard_ids)
        
        # Get chain starting from middle shard
        chain = self.manager.get_sequential_chain("shard_002")
        
        assert chain == ["shard_002", "shard_003", "shard_004"]
    
    def test_validate_metadata_integrity_valid(self):
        """Test validation of valid metadata structure."""
        # Create valid structure
        parent_metadata = self.manager.create_shard_metadata(
            "parent_001", None, DataType.DICT, ["root"]
        )
        child_metadata = self.manager.create_shard_metadata(
            "child_001", None, DataType.DICT, ["root", "child"]
        )
        
        self.manager.establish_parent_child_relationship("parent_001", "child_001")
        
        validation = self.manager.validate_metadata_integrity()
        
        assert validation["is_valid"]
        assert len(validation["errors"]) == 0
        assert validation["statistics"]["parent_child_relationships"] == 1
    
    def test_validate_metadata_integrity_broken_parent_link(self):
        """Test validation with broken parent link."""
        # Create child with non-existent parent
        child_metadata = self.manager.create_shard_metadata(
            "child_001", "nonexistent_parent", DataType.DICT, ["root", "child"]
        )
        
        validation = self.manager.validate_metadata_integrity()
        
        assert not validation["is_valid"]
        assert len(validation["errors"]) > 0
        assert any("non-existent parent" in error for error in validation["errors"])
    
    def test_validate_metadata_integrity_broken_child_link(self):
        """Test validation with broken child link."""
        # Create parent with non-existent child
        parent_metadata = self.manager.create_shard_metadata(
            "parent_001", None, DataType.DICT, ["root"], ["nonexistent_child"]
        )
        
        validation = self.manager.validate_metadata_integrity()
        
        assert not validation["is_valid"]
        assert len(validation["errors"]) > 0
        assert any("non-existent child" in error for error in validation["errors"])
    
    def test_validate_metadata_integrity_broken_sequential_link(self):
        """Test validation with broken sequential link."""
        # Create shard with non-existent next shard
        metadata = self.manager.create_shard_metadata(
            "shard_001", None, DataType.LIST, ["item"]
        )
        metadata.next_shard = "nonexistent_next"
        
        validation = self.manager.validate_metadata_integrity()
        
        assert not validation["is_valid"]
        assert len(validation["errors"]) > 0
        assert any("non-existent next shard" in error for error in validation["errors"])
    
    def test_generate_metadata_summary_empty(self):
        """Test generating summary for empty registry."""
        summary = self.manager.generate_metadata_summary()
        
        assert summary["total_shards"] == 0
        assert summary["data_type_distribution"] == {}
        assert summary["depth_distribution"] == {}
    
    def test_generate_metadata_summary_with_data(self):
        """Test generating summary with data."""
        # Create various shards
        self.manager.create_shard_metadata(
            "shard_001", None, DataType.DICT, []  # depth 0
        )
        self.manager.create_shard_metadata(
            "shard_002", "shard_001", DataType.LIST, ["level1"]  # depth 1
        )
        self.manager.create_shard_metadata(
            "shard_003", "shard_002", DataType.PRIMITIVE, ["level1", "level2"]  # depth 2
        )
        
        self.manager.establish_parent_child_relationship("shard_001", "shard_002")
        self.manager.establish_parent_child_relationship("shard_002", "shard_003")
        
        summary = self.manager.generate_metadata_summary()
        
        assert summary["total_shards"] == 3
        assert summary["data_type_distribution"]["dict"] == 1
        assert summary["data_type_distribution"]["list"] == 1
        assert summary["data_type_distribution"]["primitive"] == 1
        assert summary["depth_distribution"][0] == 1  # root
        assert summary["depth_distribution"][1] == 1  # level1
        assert summary["depth_distribution"][2] == 1  # level2
        assert summary["relationship_statistics"]["root_shards"] == 1
        assert summary["relationship_statistics"]["leaf_shards"] == 1
        assert summary["relationship_statistics"]["max_depth"] == 2
    
    def test_export_metadata_graph(self):
        """Test exporting metadata graph."""
        # Create structure with relationships
        parent_metadata = self.manager.create_shard_metadata(
            "parent_001", None, DataType.DICT, ["root"]
        )
        child_metadata = self.manager.create_shard_metadata(
            "child_001", None, DataType.LIST, ["root", "child"]
        )
        
        self.manager.establish_parent_child_relationship("parent_001", "child_001")
        self.manager.establish_sequential_linking(["parent_001", "child_001"])
        
        graph = self.manager.export_metadata_graph()
        
        assert "nodes" in graph
        assert "edges" in graph
        assert "parent_child" in graph["edges"]
        assert "sequential" in graph["edges"]
        
        # Check nodes
        assert "parent_001" in graph["nodes"]
        assert "child_001" in graph["nodes"]
        assert graph["nodes"]["parent_001"]["data_type"] == "dict"
        assert graph["nodes"]["child_001"]["data_type"] == "list"
        
        # Check edges
        assert len(graph["edges"]["parent_child"]) == 1
        assert len(graph["edges"]["sequential"]) == 1
    
    def test_detect_circular_references_no_cycles(self):
        """Test circular reference detection with no cycles."""
        # Create linear structure
        self.manager.create_shard_metadata("shard_001", None, DataType.DICT, ["root"])
        self.manager.create_shard_metadata("shard_002", None, DataType.DICT, ["root", "child"])
        
        self.manager.establish_parent_child_relationship("shard_001", "shard_002")
        
        circular_refs = self.manager._detect_circular_references()
        
        assert len(circular_refs) == 0
    
    def test_detect_circular_references_with_cycle(self):
        """Test circular reference detection with cycles."""
        # Create circular structure (this is artificial since normal operation wouldn't create this)
        metadata1 = self.manager.create_shard_metadata("shard_001", None, DataType.DICT, ["root"])
        metadata2 = self.manager.create_shard_metadata("shard_002", None, DataType.DICT, ["root", "child"])
        
        # Manually create circular reference
        metadata1.child_ids = ["shard_002"]
        metadata2.child_ids = ["shard_001"]  # This creates the cycle
        
        circular_refs = self.manager._detect_circular_references()
        
        assert len(circular_refs) > 0
    
    def test_optimize_metadata_structure_simple(self):
        """Test metadata structure optimization for simple structure."""
        # Create simple structure
        self.manager.create_shard_metadata("shard_001", None, DataType.DICT, ["root"])
        self.manager.create_shard_metadata("shard_002", "shard_001", DataType.DICT, ["root", "child"])
        
        self.manager.establish_parent_child_relationship("shard_001", "shard_002")
        
        optimization = self.manager.optimize_metadata_structure()
        
        assert "current_structure" in optimization
        assert "optimization_suggestions" in optimization
        assert "efficiency_score" in optimization
        assert 0.0 <= optimization["efficiency_score"] <= 1.0
    
    def test_optimize_metadata_structure_deep_nesting(self):
        """Test optimization suggestions for deep nesting."""
        # Create deeply nested structure
        current_parent = None
        for i in range(15):  # Create 15 levels
            shard_id = f"shard_{i:03d}"
            path = [f"level_{j}" for j in range(i)]
            
            self.manager.create_shard_metadata(shard_id, current_parent, DataType.DICT, path)
            
            if current_parent:
                self.manager.establish_parent_child_relationship(current_parent, shard_id)
            
            current_parent = shard_id
        
        optimization = self.manager.optimize_metadata_structure()
        
        # Should suggest reducing nesting
        suggestions = optimization["optimization_suggestions"]
        assert any(s["type"] == "reduce_nesting" for s in suggestions)
    
    def test_optimize_metadata_structure_many_roots(self):
        """Test optimization suggestions for many root shards."""
        # Create many root shards
        for i in range(10):
            self.manager.create_shard_metadata(f"root_{i}", None, DataType.DICT, [f"root_{i}"])
        
        optimization = self.manager.optimize_metadata_structure()
        
        # Should suggest consolidating roots
        suggestions = optimization["optimization_suggestions"]
        assert any(s["type"] == "consolidate_roots" for s in suggestions)
    
    def test_calculate_efficiency_score_perfect(self):
        """Test efficiency score calculation for perfect structure."""
        # Create balanced structure
        self.manager.create_shard_metadata("root", None, DataType.DICT, [])
        self.manager.create_shard_metadata("child1", "root", DataType.DICT, ["child1"])
        self.manager.create_shard_metadata("child2", "root", DataType.DICT, ["child2"])
        
        self.manager.establish_parent_child_relationship("root", "child1")
        self.manager.establish_parent_child_relationship("root", "child2")
        
        summary = self.manager.generate_metadata_summary()
        score = self.manager._calculate_efficiency_score(summary)
        
        assert score == 1.0  # Perfect score
    
    def test_clear_registry(self):
        """Test clearing the metadata registry."""
        # Add some metadata
        self.manager.create_shard_metadata("shard_001", None, DataType.DICT, ["root"])
        
        assert len(self.manager._shard_registry) == 1
        
        # Clear registry
        self.manager.clear_registry()
        
        assert len(self.manager._shard_registry) == 0
        assert len(self.manager._relationship_graph) == 0
    
    def test_get_shard_metadata_nonexistent(self):
        """Test getting metadata for non-existent shard."""
        metadata = self.manager.get_shard_metadata("nonexistent_shard")
        
        assert metadata is None