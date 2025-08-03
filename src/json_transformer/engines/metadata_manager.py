"""Metadata manager for generating and managing shard metadata."""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from ..types import DataType, DataStructure
from ..models import ShardMetadata, GlobalMetadata, FileShard
from ..utils.size_calculator import SizeCalculator


class MetadataManager:
    """
    Manager for generating shard metadata and maintaining relationships.
    
    Handles the creation of metadata for shards including parent-child
    relationship tracking and sequential linking for ordered data.
    """
    
    def __init__(self, size_calculator: Optional[SizeCalculator] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the metadata manager.
        
        Args:
            size_calculator: Optional SizeCalculator instance
            logger: Optional logger instance
        """
        self.size_calculator = size_calculator or SizeCalculator()
        self.logger = logger or logging.getLogger(__name__)
        self._shard_registry: Dict[str, ShardMetadata] = {}
        self._relationship_graph: Dict[str, Set[str]] = {}
    
    def create_shard_metadata(self, shard_id: str, parent_id: Optional[str],
                             data_type: DataType, original_path: List[str],
                             child_ids: Optional[List[str]] = None) -> ShardMetadata:
        """
        Create metadata for a shard.
        
        Args:
            shard_id: Unique identifier for the shard
            parent_id: ID of parent shard (if any)
            data_type: Type of data in the shard
            original_path: Path to this data in the original structure
            child_ids: List of child shard IDs
            
        Returns:
            ShardMetadata instance
        """
        if child_ids is None:
            child_ids = []
        
        metadata = ShardMetadata(
            shard_id=shard_id,
            parent_id=parent_id,
            child_ids=child_ids.copy(),
            data_type=data_type,
            original_path=original_path.copy()
        )
        
        # Register the metadata
        self._register_shard_metadata(metadata)
        
        self.logger.debug(f"Created metadata for shard {shard_id}")
        return metadata
    
    def create_global_metadata(self, version: str, original_size: int,
                              shard_count: int, root_shard: str,
                              data_structure: DataStructure) -> GlobalMetadata:
        """
        Create global metadata for the transformation.
        
        Args:
            version: Version of the transformation format
            original_size: Size of original data in bytes
            shard_count: Total number of shards
            root_shard: ID of the root shard
            data_structure: Type of data structure
            
        Returns:
            GlobalMetadata instance
        """
        global_metadata = GlobalMetadata(
            version=version,
            original_size=original_size,
            shard_count=shard_count,
            root_shard=root_shard,
            created_at=datetime.now(),
            data_structure=data_structure
        )
        
        self.logger.info(f"Created global metadata for {shard_count} shards")
        return global_metadata
    
    def establish_parent_child_relationship(self, parent_id: str, child_id: str) -> None:
        """
        Establish parent-child relationship between shards.
        
        Args:
            parent_id: ID of parent shard
            child_id: ID of child shard
        """
        # Update parent's child list
        if parent_id in self._shard_registry:
            parent_metadata = self._shard_registry[parent_id]
            if child_id not in parent_metadata.child_ids:
                parent_metadata.add_child(child_id)
        
        # Update child's parent
        if child_id in self._shard_registry:
            child_metadata = self._shard_registry[child_id]
            child_metadata.parent_id = parent_id
        
        # Update relationship graph
        if parent_id not in self._relationship_graph:
            self._relationship_graph[parent_id] = set()
        self._relationship_graph[parent_id].add(child_id)
        
        self.logger.debug(f"Established parent-child relationship: {parent_id} -> {child_id}")
    
    def establish_sequential_linking(self, shard_ids: List[str]) -> None:
        """
        Establish sequential linking between shards.
        
        Args:
            shard_ids: List of shard IDs in sequential order
        """
        for i, shard_id in enumerate(shard_ids):
            if shard_id not in self._shard_registry:
                self.logger.warning(f"Shard {shard_id} not found in registry")
                continue
            
            metadata = self._shard_registry[shard_id]
            
            # Set previous shard
            if i > 0:
                metadata.previous_shard = shard_ids[i - 1]
            else:
                metadata.previous_shard = None
            
            # Set next shard
            if i < len(shard_ids) - 1:
                metadata.next_shard = shard_ids[i + 1]
            else:
                metadata.next_shard = None
        
        self.logger.debug(f"Established sequential linking for {len(shard_ids)} shards")
    
    def update_shard_metadata(self, shard_id: str, **updates) -> bool:
        """
        Update metadata for a shard.
        
        Args:
            shard_id: ID of shard to update
            **updates: Metadata fields to update
            
        Returns:
            True if update was successful
        """
        if shard_id not in self._shard_registry:
            self.logger.error(f"Shard {shard_id} not found in registry")
            return False
        
        metadata = self._shard_registry[shard_id]
        
        # Update allowed fields
        allowed_updates = {
            'parent_id', 'child_ids', 'data_type', 'original_path',
            'next_shard', 'previous_shard'
        }
        
        for field, value in updates.items():
            if field in allowed_updates:
                setattr(metadata, field, value)
                self.logger.debug(f"Updated {field} for shard {shard_id}")
            else:
                self.logger.warning(f"Ignoring invalid update field: {field}")
        
        return True
    
    def get_shard_metadata(self, shard_id: str) -> Optional[ShardMetadata]:
        """
        Get metadata for a shard.
        
        Args:
            shard_id: ID of shard
            
        Returns:
            ShardMetadata instance or None if not found
        """
        return self._shard_registry.get(shard_id)
    
    def get_children_metadata(self, parent_id: str) -> List[ShardMetadata]:
        """
        Get metadata for all children of a parent shard.
        
        Args:
            parent_id: ID of parent shard
            
        Returns:
            List of ShardMetadata instances
        """
        if parent_id not in self._shard_registry:
            return []
        
        parent_metadata = self._shard_registry[parent_id]
        children_metadata = []
        
        for child_id in parent_metadata.child_ids:
            if child_id in self._shard_registry:
                children_metadata.append(self._shard_registry[child_id])
        
        return children_metadata
    
    def get_sequential_chain(self, start_shard_id: str) -> List[str]:
        """
        Get the sequential chain starting from a shard.
        
        Args:
            start_shard_id: ID of starting shard
            
        Returns:
            List of shard IDs in sequential order
        """
        chain = []
        current_id = start_shard_id
        visited = set()
        
        while current_id and current_id not in visited:
            chain.append(current_id)
            visited.add(current_id)
            
            if current_id in self._shard_registry:
                metadata = self._shard_registry[current_id]
                current_id = metadata.next_shard
            else:
                break
        
        return chain
    
    def validate_metadata_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of all metadata relationships.
        
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {
                "total_shards": len(self._shard_registry),
                "parent_child_relationships": 0,
                "sequential_links": 0,
                "orphaned_shards": 0,
                "circular_references": 0
            }
        }
        
        # Check parent-child relationships
        for shard_id, metadata in self._shard_registry.items():
            # Validate parent relationship
            if metadata.parent_id:
                if metadata.parent_id not in self._shard_registry:
                    validation_result["errors"].append(
                        f"Shard {shard_id} has non-existent parent {metadata.parent_id}"
                    )
                    validation_result["is_valid"] = False
                else:
                    parent_metadata = self._shard_registry[metadata.parent_id]
                    if shard_id not in parent_metadata.child_ids:
                        validation_result["errors"].append(
                            f"Broken parent-child link: {metadata.parent_id} -> {shard_id}"
                        )
                        validation_result["is_valid"] = False
                    else:
                        validation_result["statistics"]["parent_child_relationships"] += 1
            
            # Validate child relationships
            for child_id in metadata.child_ids:
                if child_id not in self._shard_registry:
                    validation_result["errors"].append(
                        f"Shard {shard_id} has non-existent child {child_id}"
                    )
                    validation_result["is_valid"] = False
                else:
                    child_metadata = self._shard_registry[child_id]
                    if child_metadata.parent_id != shard_id:
                        validation_result["errors"].append(
                            f"Broken child-parent link: {shard_id} -> {child_id}"
                        )
                        validation_result["is_valid"] = False
            
            # Validate sequential links
            if metadata.next_shard:
                if metadata.next_shard not in self._shard_registry:
                    validation_result["errors"].append(
                        f"Shard {shard_id} has non-existent next shard {metadata.next_shard}"
                    )
                    validation_result["is_valid"] = False
                else:
                    next_metadata = self._shard_registry[metadata.next_shard]
                    if next_metadata.previous_shard != shard_id:
                        validation_result["errors"].append(
                            f"Broken sequential link: {shard_id} -> {metadata.next_shard}"
                        )
                        validation_result["is_valid"] = False
                    else:
                        validation_result["statistics"]["sequential_links"] += 1
            
            # Check for orphaned shards (no parent and no children)
            if not metadata.parent_id and not metadata.child_ids:
                validation_result["statistics"]["orphaned_shards"] += 1
        
        # Check for circular references
        circular_refs = self._detect_circular_references()
        validation_result["statistics"]["circular_references"] = len(circular_refs)
        
        if circular_refs:
            validation_result["errors"].extend([
                f"Circular reference detected: {' -> '.join(cycle)}"
                for cycle in circular_refs
            ])
            validation_result["is_valid"] = False
        
        return validation_result
    
    def generate_metadata_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of all metadata.
        
        Returns:
            Dictionary with metadata summary
        """
        if not self._shard_registry:
            return {
                "total_shards": 0,
                "data_type_distribution": {},
                "depth_distribution": {},
                "relationship_statistics": {}
            }
        
        # Count data types
        data_type_counts = {}
        depth_counts = {}
        
        for metadata in self._shard_registry.values():
            # Count data types
            data_type = metadata.data_type.value
            data_type_counts[data_type] = data_type_counts.get(data_type, 0) + 1
            
            # Count depths
            depth = len(metadata.original_path)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        
        # Calculate relationship statistics
        root_shards = [m for m in self._shard_registry.values() if not m.parent_id]
        leaf_shards = [m for m in self._shard_registry.values() if not m.child_ids]
        
        # Find longest sequential chain
        longest_chain = 0
        for shard_id in self._shard_registry:
            chain = self.get_sequential_chain(shard_id)
            longest_chain = max(longest_chain, len(chain))
        
        return {
            "total_shards": len(self._shard_registry),
            "data_type_distribution": data_type_counts,
            "depth_distribution": depth_counts,
            "relationship_statistics": {
                "root_shards": len(root_shards),
                "leaf_shards": len(leaf_shards),
                "max_depth": max(depth_counts.keys()) if depth_counts else 0,
                "longest_sequential_chain": longest_chain,
                "total_relationships": len(self._relationship_graph)
            }
        }
    
    def export_metadata_graph(self) -> Dict[str, Any]:
        """
        Export the metadata relationship graph.
        
        Returns:
            Dictionary representing the metadata graph
        """
        graph = {
            "nodes": {},
            "edges": {
                "parent_child": [],
                "sequential": []
            }
        }
        
        # Add nodes
        for shard_id, metadata in self._shard_registry.items():
            graph["nodes"][shard_id] = {
                "data_type": metadata.data_type.value,
                "original_path": metadata.original_path,
                "is_root": metadata.is_root(),
                "is_leaf": metadata.is_leaf()
            }
        
        # Add parent-child edges
        for shard_id, metadata in self._shard_registry.items():
            for child_id in metadata.child_ids:
                graph["edges"]["parent_child"].append({
                    "from": shard_id,
                    "to": child_id,
                    "type": "parent_child"
                })
        
        # Add sequential edges
        for shard_id, metadata in self._shard_registry.items():
            if metadata.next_shard:
                graph["edges"]["sequential"].append({
                    "from": shard_id,
                    "to": metadata.next_shard,
                    "type": "sequential"
                })
        
        return graph
    
    def _register_shard_metadata(self, metadata: ShardMetadata) -> None:
        """
        Register shard metadata in the internal registry.
        
        Args:
            metadata: ShardMetadata to register
        """
        self._shard_registry[metadata.shard_id] = metadata
    
    def _detect_circular_references(self) -> List[List[str]]:
        """
        Detect circular references in the relationship graph.
        
        Returns:
            List of circular reference chains
        """
        circular_refs = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> None:
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                circular_refs.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            # Check children
            if node in self._shard_registry:
                metadata = self._shard_registry[node]
                for child_id in metadata.child_ids:
                    if child_id in self._shard_registry:
                        dfs(child_id, path + [node])
            
            rec_stack.remove(node)
        
        # Check all nodes
        for shard_id in self._shard_registry:
            if shard_id not in visited:
                dfs(shard_id, [])
        
        return circular_refs
    
    def optimize_metadata_structure(self) -> Dict[str, Any]:
        """
        Analyze and suggest optimizations for the metadata structure.
        
        Returns:
            Dictionary with optimization suggestions
        """
        summary = self.generate_metadata_summary()
        suggestions = []
        
        # Check for deep nesting
        max_depth = summary["relationship_statistics"]["max_depth"]
        if max_depth > 10:
            suggestions.append({
                "type": "reduce_nesting",
                "description": f"Very deep nesting detected (depth: {max_depth}). Consider flattening structure.",
                "priority": "high"
            })
        
        # Check for too many root shards
        root_count = summary["relationship_statistics"]["root_shards"]
        if root_count > 5:
            suggestions.append({
                "type": "consolidate_roots",
                "description": f"Many root shards ({root_count}). Consider creating a single root.",
                "priority": "medium"
            })
        
        # Check for unbalanced tree
        total_shards = summary["total_shards"]
        leaf_count = summary["relationship_statistics"]["leaf_shards"]
        if leaf_count > total_shards * 0.8:  # More than 80% are leaves
            suggestions.append({
                "type": "balance_tree",
                "description": "Tree structure is very flat. Consider grouping related shards.",
                "priority": "low"
            })
        
        # Check for long sequential chains
        longest_chain = summary["relationship_statistics"]["longest_sequential_chain"]
        if longest_chain > 20:
            suggestions.append({
                "type": "break_long_chains",
                "description": f"Very long sequential chain ({longest_chain}). Consider breaking into smaller chunks.",
                "priority": "medium"
            })
        
        return {
            "current_structure": summary,
            "optimization_suggestions": suggestions,
            "efficiency_score": self._calculate_efficiency_score(summary)
        }
    
    def _calculate_efficiency_score(self, summary: Dict[str, Any]) -> float:
        """
        Calculate efficiency score for the metadata structure.
        
        Args:
            summary: Metadata summary
            
        Returns:
            Efficiency score (0.0 to 1.0)
        """
        score = 1.0
        
        # Penalize deep nesting
        max_depth = summary["relationship_statistics"]["max_depth"]
        if max_depth > 5:
            score -= (max_depth - 5) * 0.05
        
        # Penalize too many roots
        root_count = summary["relationship_statistics"]["root_shards"]
        if root_count > 1:
            score -= (root_count - 1) * 0.1
        
        # Penalize very flat structures
        total_shards = summary["total_shards"]
        leaf_count = summary["relationship_statistics"]["leaf_shards"]
        if total_shards > 0:
            leaf_ratio = leaf_count / total_shards
            if leaf_ratio > 0.8:
                score -= (leaf_ratio - 0.8) * 0.5
        
        return max(0.0, min(1.0, score))
    
    def clear_registry(self) -> None:
        """Clear the metadata registry."""
        self._shard_registry.clear()
        self._relationship_graph.clear()
        self.logger.debug("Cleared metadata registry")