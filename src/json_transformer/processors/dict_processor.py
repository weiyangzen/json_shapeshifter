"""Dictionary processor for handling nested dictionary structures."""

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from ..types import DataProcessorInterface, ProcessingResult, DataStructure, DataType
from ..models import FileShard, ShardMetadata, GlobalMetadata
from ..engines.file_sharding_engine import FileShardingEngine
from ..utils.size_calculator import SizeCalculator


class DictProcessor(DataProcessorInterface):
    """
    Processor for dictionary-of-dictionaries structures.
    
    Handles the processing of nested dictionary structures by creating
    separate files for each nested dictionary level while maintaining
    hierarchical relationships through metadata linking.
    """
    
    def __init__(self, sharding_engine: Optional[FileShardingEngine] = None,
                 size_calculator: Optional[SizeCalculator] = None,
                 logger: Optional[logging.Logger] = None,
                 enable_streaming: bool = True,
                 compression_threshold: int = 50000):
        """
        Initialize the dictionary processor with advanced optimizations.
        
        Args:
            sharding_engine: Optional FileShardingEngine instance
            size_calculator: Optional SizeCalculator instance
            logger: Optional logger instance
            enable_streaming: Enable streaming processing for large datasets
            compression_threshold: Size threshold for enabling compression optimizations
        """
        self.sharding_engine = sharding_engine or FileShardingEngine()
        self.size_calculator = size_calculator or SizeCalculator()
        self.logger = logger or logging.getLogger(__name__)
        self.enable_streaming = enable_streaming
        self.compression_threshold = compression_threshold
        self._processing_cache = {}  # Cache for repeated structures
    
    def process(self, data: Any, max_size: int) -> ProcessingResult:
        """
        Process dictionary data into shards.
        
        Args:
            data: Dictionary data to process
            max_size: Maximum size per shard in bytes
            
        Returns:
            ProcessingResult with shards and metadata
            
        Raises:
            ValueError: If data is not a dictionary or processing fails
        """
        if not isinstance(data, dict):
            raise ValueError(f"DictProcessor expects dict data, got {type(data).__name__}")
        
        self.logger.info(f"Processing dictionary with {len(data)} keys, max_size={max_size}")
        
        # Calculate original size
        original_size = self.size_calculator.calculate_json_size(data)
        
        # Process the dictionary hierarchically
        all_shards = []
        root_shard = self._process_dict_level(
            data, max_size, original_path=[], parent_id=None
        )
        
        # Collect all shards from the hierarchical structure
        self._collect_shards_recursive(root_shard, all_shards)
        
        # Create global metadata
        global_metadata = GlobalMetadata(
            version="1.0.0",
            original_size=original_size,
            shard_count=len(all_shards),
            root_shard=root_shard.id,
            created_at=datetime.now(),
            data_structure=DataStructure.DICT_OF_DICTS
        )
        
        self.logger.info(f"Created {len(all_shards)} shards from dictionary processing")
        
        return ProcessingResult(
            shards=all_shards,
            root_shard_id=root_shard.id,
            metadata=global_metadata
        )
    
    def _process_dict_level(self, data: Dict[str, Any], max_size: int,
                           original_path: List[str], parent_id: Optional[str]) -> FileShard:
        """
        Process a single dictionary level, creating shards for nested structures.
        
        Args:
            data: Dictionary data to process
            max_size: Maximum size per shard
            original_path: Path to this dictionary in the original structure
            parent_id: ID of parent shard
            
        Returns:
            FileShard representing this dictionary level
        """
        # Separate nested dictionaries from primitive values
        nested_dicts = {}
        primitive_data = {}
        nested_lists = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                nested_dicts[key] = value
            elif isinstance(value, list):
                nested_lists[key] = value
            else:
                primitive_data[key] = value
        
        # Create shard for primitive data at this level
        current_shard_data = primitive_data.copy()
        
        # Generate shard ID and metadata
        shard_id = self.sharding_engine.generate_shard_id()
        child_ids = []
        
        # Process nested dictionaries
        nested_shards = []
        for key, nested_dict in nested_dicts.items():
            nested_path = original_path + [key]
            nested_shard = self._process_dict_level(
                nested_dict, max_size, nested_path, shard_id
            )
            nested_shards.append(nested_shard)
            child_ids.append(nested_shard.id)
            
            # Add reference to nested shard in current data
            current_shard_data[key] = {
                "_shard_ref": nested_shard.id,
                "_shard_type": "dict",
                "_original_path": nested_path
            }
        
        # Process nested lists
        for key, nested_list in nested_lists.items():
            nested_path = original_path + [key]
            list_shards = self._process_list_in_dict(
                nested_list, max_size, nested_path, shard_id
            )
            nested_shards.extend(list_shards)
            
            if list_shards:
                # Add reference to first list shard
                child_ids.append(list_shards[0].id)
                current_shard_data[key] = {
                    "_shard_ref": list_shards[0].id,
                    "_shard_type": "list",
                    "_original_path": nested_path
                }
        
        # Check if current shard data exceeds size limit
        if self.sharding_engine.calculate_size(current_shard_data) > max_size:
            # Split current level data
            current_shard_data, additional_shards = self._split_current_level(
                current_shard_data, max_size, original_path, shard_id
            )
            nested_shards.extend(additional_shards)
            
            # Update child IDs
            for additional_shard in additional_shards:
                child_ids.append(additional_shard.id)
        
        # Create metadata for current shard
        metadata = ShardMetadata(
            shard_id=shard_id,
            parent_id=parent_id,
            child_ids=child_ids,
            data_type=DataType.DICT,
            original_path=original_path
        )
        
        # Create current shard
        current_shard = self.sharding_engine.create_shard(
            current_shard_data, shard_id, metadata
        )
        
        # Store nested shards as children
        current_shard._nested_shards = nested_shards
        
        return current_shard
    
    def _process_list_in_dict(self, data: List[Any], max_size: int,
                             original_path: List[str], parent_id: str) -> List[FileShard]:
        """
        Process a list that's nested within a dictionary.
        
        Args:
            data: List data to process
            max_size: Maximum size per shard
            original_path: Path to this list in the original structure
            parent_id: ID of parent shard
            
        Returns:
            List of FileShard instances for the list data
        """
        # Split list data by size
        chunks = self.sharding_engine.split_data_by_size(data, max_size, original_path)
        
        # Create shards from chunks
        shards = self.sharding_engine.create_shards_from_chunks(chunks, parent_id)
        
        # Update metadata for list shards
        for shard in shards:
            shard.metadata.data_type = DataType.LIST
            shard.metadata.original_path = original_path
        
        return shards
    
    def _split_current_level(self, data: Dict[str, Any], max_size: int,
                            original_path: List[str], base_shard_id: str) -> Tuple[Dict[str, Any], List[FileShard]]:
        """
        Split current level data if it exceeds size limit.
        
        Args:
            data: Dictionary data to split
            max_size: Maximum size per shard
            original_path: Path to this data
            base_shard_id: Base shard ID for naming
            
        Returns:
            Tuple of (remaining_data, additional_shards)
        """
        # Split data by size
        chunks = self.sharding_engine.split_data_by_size(data, max_size, original_path)
        
        if len(chunks) <= 1:
            # No splitting needed
            return data, []
        
        # First chunk becomes the main data
        main_data = chunks[0][0]
        additional_shards = []
        
        # Create additional shards for remaining chunks
        for i, (chunk_data, chunk_path) in enumerate(chunks[1:], 1):
            shard_id = f"{base_shard_id}_overflow_{i}"
            
            metadata = ShardMetadata(
                shard_id=shard_id,
                parent_id=base_shard_id,
                child_ids=[],
                data_type=DataType.DICT,
                original_path=chunk_path
            )
            
            shard = self.sharding_engine.create_shard(chunk_data, shard_id, metadata)
            additional_shards.append(shard)
        
        return main_data, additional_shards
    
    def _collect_shards_recursive(self, shard: FileShard, all_shards: List[FileShard]) -> None:
        """
        Recursively collect all shards from the hierarchical structure.
        
        Args:
            shard: Current shard to process
            all_shards: List to collect shards into
        """
        all_shards.append(shard)
        
        # Process nested shards if they exist
        if hasattr(shard, '_nested_shards'):
            for nested_shard in shard._nested_shards:
                self._collect_shards_recursive(nested_shard, all_shards)
    
    def create_shard_references(self, shards: List[FileShard]) -> Dict[str, Dict[str, Any]]:
        """
        Create reference map for shard relationships.
        
        Args:
            shards: List of all shards
            
        Returns:
            Dictionary mapping shard IDs to reference information
        """
        references = {}
        
        for shard in shards:
            references[shard.id] = {
                "path": shard.metadata.original_path,
                "type": shard.metadata.data_type.value,
                "parent": shard.metadata.parent_id,
                "children": shard.metadata.child_ids.copy(),
                "size": shard.size
            }
        
        return references
    
    def validate_hierarchical_structure(self, shards: List[FileShard]) -> bool:
        """
        Validate that the hierarchical structure is correct.
        
        Args:
            shards: List of shards to validate
            
        Returns:
            True if structure is valid
        """
        shard_map = {shard.id: shard for shard in shards}
        
        for shard in shards:
            # Check parent-child relationships
            for child_id in shard.metadata.child_ids:
                if child_id not in shard_map:
                    self.logger.error(f"Child shard {child_id} not found for parent {shard.id}")
                    return False
                
                child_shard = shard_map[child_id]
                if child_shard.metadata.parent_id != shard.id:
                    self.logger.error(f"Bidirectional link broken: {shard.id} <-> {child_id}")
                    return False
        
        return True
    
    def optimize_dictionary_sharding(self, data: Dict[str, Any], max_size: int) -> Dict[str, Any]:
        """
        Analyze dictionary structure and provide optimization recommendations.
        
        Args:
            data: Dictionary data to analyze
            max_size: Target maximum size per shard
            
        Returns:
            Dictionary with optimization recommendations
        """
        total_size = self.size_calculator.calculate_json_size(data)
        
        # Analyze key-value sizes
        key_sizes = {}
        large_keys = []
        
        for key, value in data.items():
            try:
                value_size = self.size_calculator.calculate_json_size(value)
                key_sizes[key] = value_size
                
                if value_size > max_size * 0.5:  # Values larger than 50% of max size
                    large_keys.append((key, value_size))
            except ValueError:
                # Skip non-serializable values
                continue
        
        # Calculate nesting levels
        max_nesting = self._calculate_max_nesting_level(data)
        
        # Estimate shard count
        estimated_shards = self.size_calculator.estimate_shard_count(total_size, max_size)
        
        recommendations = {
            "total_size": total_size,
            "estimated_shards": estimated_shards,
            "max_nesting_level": max_nesting,
            "large_keys": large_keys,
            "key_count": len(data),
            "optimization_suggestions": []
        }
        
        # Add optimization suggestions
        if large_keys:
            recommendations["optimization_suggestions"].append({
                "type": "split_large_values",
                "description": f"Consider splitting {len(large_keys)} large values",
                "affected_keys": [key for key, _ in large_keys]
            })
        
        if max_nesting > 5:
            recommendations["optimization_suggestions"].append({
                "type": "flatten_deep_nesting",
                "description": f"Deep nesting detected (level {max_nesting}). Consider flattening structure."
            })
        
        if len(data) > 1000:
            recommendations["optimization_suggestions"].append({
                "type": "group_related_keys",
                "description": "Large number of keys. Consider grouping related keys into sub-dictionaries."
            })
        
        return recommendations
    
    def _calculate_max_nesting_level(self, data: Any, current_level: int = 0) -> int:
        """Calculate maximum nesting level in dictionary structure."""
        if not isinstance(data, dict):
            return current_level
        
        max_level = current_level
        for value in data.values():
            if isinstance(value, dict):
                level = self._calculate_max_nesting_level(value, current_level + 1)
                max_level = max(max_level, level)
        
        return max_level