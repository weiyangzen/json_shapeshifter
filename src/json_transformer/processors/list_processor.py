"""List processor for handling list/array data structures."""

import logging
from typing import Any, List, Optional, Tuple
from datetime import datetime
from ..types import DataProcessorInterface, ProcessingResult, DataStructure, DataType
from ..models import FileShard, ShardMetadata, GlobalMetadata
from ..engines.file_sharding_engine import FileShardingEngine
from ..utils.size_calculator import SizeCalculator


class ListProcessor(DataProcessorInterface):
    """
    Processor for list/array data structures.
    
    Handles the processing of list structures by splitting large lists
    while maintaining order through sequential linking for list elements.
    """
    
    def __init__(self, sharding_engine: Optional[FileShardingEngine] = None,
                 size_calculator: Optional[SizeCalculator] = None,
                 logger: Optional[logging.Logger] = None,
                 enable_streaming: bool = True,
                 batch_size: int = 1000):
        """
        Initialize the list processor with advanced optimizations.
        
        Args:
            sharding_engine: Optional FileShardingEngine instance
            size_calculator: Optional SizeCalculator instance
            logger: Optional logger instance
            enable_streaming: Enable streaming processing for large lists
            batch_size: Batch size for processing large lists efficiently
        """
        self.sharding_engine = sharding_engine or FileShardingEngine()
        self.size_calculator = size_calculator or SizeCalculator()
        self.logger = logger or logging.getLogger(__name__)
        self.enable_streaming = enable_streaming
        self.batch_size = batch_size
        self._item_size_cache = {}  # Cache for item size calculations
    
    def process(self, data: Any, max_size: int) -> ProcessingResult:
        """
        Process list data into shards.
        
        Args:
            data: List data to process
            max_size: Maximum size per shard in bytes
            
        Returns:
            ProcessingResult with shards and metadata
            
        Raises:
            ValueError: If data is not a list or processing fails
        """
        if not isinstance(data, list):
            raise ValueError(f"ListProcessor expects list data, got {type(data).__name__}")
        
        self.logger.info(f"Processing list with {len(data)} items, max_size={max_size}")
        
        # Calculate original size
        original_size = self.size_calculator.calculate_json_size(data)
        
        # Process the list
        shards = self._process_list_data(data, max_size)
        
        # Ensure we have at least one shard
        if not shards:
            # Create empty shard for empty list
            empty_shard = self._create_empty_list_shard()
            shards = [empty_shard]
        
        # Create global metadata
        global_metadata = GlobalMetadata(
            version="1.0.0",
            original_size=original_size,
            shard_count=len(shards),
            root_shard=shards[0].id,
            created_at=datetime.now(),
            data_structure=DataStructure.LIST
        )
        
        self.logger.info(f"Created {len(shards)} shards from list processing")
        
        return ProcessingResult(
            shards=shards,
            root_shard_id=shards[0].id,
            metadata=global_metadata
        )
    
    def _process_list_data(self, data: List[Any], max_size: int) -> List[FileShard]:
        """
        Process list data into appropriately sized shards.
        
        Args:
            data: List data to process
            max_size: Maximum size per shard
            
        Returns:
            List of FileShard instances
        """
        if not data:
            return []
        
        # Analyze list structure to determine processing strategy
        analysis = self._analyze_list_structure(data)
        
        if analysis["has_nested_structures"]:
            return self._process_list_with_nested_structures(data, max_size, analysis)
        else:
            return self._process_simple_list(data, max_size)
    
    def _analyze_list_structure(self, data: List[Any]) -> dict:
        """
        Analyze the structure of list data.
        
        Args:
            data: List data to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not data:
            return {
                "has_nested_structures": False,
                "item_types": {},
                "max_item_size": 0,
                "homogeneous": True
            }
        
        item_types = {}
        max_item_size = 0
        has_nested = False
        
        for item in data:
            item_type = type(item).__name__
            item_types[item_type] = item_types.get(item_type, 0) + 1
            
            # Check if item is nested structure
            if isinstance(item, (dict, list)):
                has_nested = True
            
            # Calculate item size
            try:
                item_size = self.size_calculator.calculate_json_size(item)
                max_item_size = max(max_item_size, item_size)
            except ValueError:
                # Skip non-serializable items
                continue
        
        return {
            "has_nested_structures": has_nested,
            "item_types": item_types,
            "max_item_size": max_item_size,
            "homogeneous": len(item_types) == 1,
            "total_items": len(data)
        }
    
    def _process_simple_list(self, data: List[Any], max_size: int) -> List[FileShard]:
        """
        Process a simple list without nested structures.
        
        Args:
            data: List data to process
            max_size: Maximum size per shard
            
        Returns:
            List of FileShard instances
        """
        # Split list into chunks based on size
        chunks = self._split_list_by_size(data, max_size)
        
        # Create shards from chunks
        shards = []
        for i, chunk in enumerate(chunks):
            shard_id = self.sharding_engine.generate_shard_id()
            
            # Create metadata with sequential linking
            metadata = ShardMetadata(
                shard_id=shard_id,
                parent_id=None,
                child_ids=[],
                data_type=DataType.LIST,
                original_path=[f"[{i * len(chunk)}:{(i + 1) * len(chunk)}]"]
            )
            
            # Set up sequential linking
            if i > 0:
                metadata.previous_shard = shards[i-1].id
                shards[i-1].metadata.next_shard = shard_id
            
            # Create shard
            shard = self.sharding_engine.create_shard(chunk, shard_id, metadata)
            shards.append(shard)
        
        return shards
    
    def _process_list_with_nested_structures(self, data: List[Any], max_size: int, 
                                           analysis: dict) -> List[FileShard]:
        """
        Process a list containing nested structures.
        
        Args:
            data: List data to process
            max_size: Maximum size per shard
            analysis: Pre-computed analysis of the list structure
            
        Returns:
            List of FileShard instances
        """
        all_shards = []
        current_chunk = []
        current_size = 0
        chunk_start_index = 0
        
        for i, item in enumerate(data):
            # Calculate item size
            try:
                item_size = self.size_calculator.calculate_json_size(item)
            except ValueError:
                # Skip non-serializable items
                self.logger.warning(f"Skipping non-serializable item at index {i}")
                continue
            
            # Check if item is too large for any shard
            if item_size > max_size:
                # Save current chunk if it has items
                if current_chunk:
                    chunk_shards = self._create_chunk_shards(
                        current_chunk, chunk_start_index, max_size
                    )
                    all_shards.extend(chunk_shards)
                    current_chunk = []
                    current_size = 0
                
                # Handle oversized item
                oversized_shards = self._handle_oversized_item(item, i, max_size)
                all_shards.extend(oversized_shards)
                
                chunk_start_index = i + 1
                continue
            
            # Check if adding this item would exceed size limit
            if self.sharding_engine.should_create_new_shard(current_size, item_size, max_size):
                # Save current chunk
                if current_chunk:
                    chunk_shards = self._create_chunk_shards(
                        current_chunk, chunk_start_index, max_size
                    )
                    all_shards.extend(chunk_shards)
                
                # Start new chunk
                current_chunk = [item]
                current_size = item_size
                chunk_start_index = i
            else:
                # Add to current chunk
                current_chunk.append(item)
                current_size += item_size
        
        # Handle final chunk
        if current_chunk:
            chunk_shards = self._create_chunk_shards(
                current_chunk, chunk_start_index, max_size
            )
            all_shards.extend(chunk_shards)
        
        # Set up sequential linking between all shards
        self._setup_sequential_linking(all_shards)
        
        return all_shards
    
    def _split_list_by_size(self, data: List[Any], max_size: int) -> List[List[Any]]:
        """
        Split list into chunks based on size constraints.
        
        Args:
            data: List data to split
            max_size: Maximum size per chunk
            
        Returns:
            List of chunks
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        for item in data:
            try:
                item_size = self.size_calculator.calculate_json_size(item)
            except ValueError:
                # Skip non-serializable items
                continue
            
            # Check if adding this item would exceed size limit
            if current_chunk and current_size + item_size > max_size * 0.9:  # 90% threshold
                # Save current chunk and start new one
                chunks.append(current_chunk)
                current_chunk = [item]
                current_size = item_size
            else:
                # Add to current chunk
                current_chunk.append(item)
                current_size += item_size
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [[]]  # Return empty chunk if no data
    
    def _create_chunk_shards(self, chunk: List[Any], start_index: int, 
                           max_size: int) -> List[FileShard]:
        """
        Create shards from a list chunk.
        
        Args:
            chunk: List chunk to create shards from
            start_index: Starting index of this chunk in the original list
            max_size: Maximum size per shard
            
        Returns:
            List of FileShard instances
        """
        # Check if chunk needs further splitting
        chunk_size = self.size_calculator.calculate_json_size(chunk)
        
        if chunk_size <= max_size:
            # Create single shard for chunk
            shard_id = self.sharding_engine.generate_shard_id()
            
            metadata = ShardMetadata(
                shard_id=shard_id,
                parent_id=None,
                child_ids=[],
                data_type=DataType.LIST,
                original_path=[f"[{start_index}:{start_index + len(chunk)}]"]
            )
            
            shard = self.sharding_engine.create_shard(chunk, shard_id, metadata)
            return [shard]
        else:
            # Further split the chunk
            sub_chunks = self._split_list_by_size(chunk, max_size)
            shards = []
            
            for i, sub_chunk in enumerate(sub_chunks):
                shard_id = self.sharding_engine.generate_shard_id()
                sub_start = start_index + sum(len(sc) for sc in sub_chunks[:i])
                
                metadata = ShardMetadata(
                    shard_id=shard_id,
                    parent_id=None,
                    child_ids=[],
                    data_type=DataType.LIST,
                    original_path=[f"[{sub_start}:{sub_start + len(sub_chunk)}]"]
                )
                
                shard = self.sharding_engine.create_shard(sub_chunk, shard_id, metadata)
                shards.append(shard)
            
            return shards
    
    def _handle_oversized_item(self, item: Any, index: int, max_size: int) -> List[FileShard]:
        """
        Handle an item that exceeds the maximum shard size.
        
        Args:
            item: Oversized item
            index: Index of the item in the original list
            max_size: Maximum size per shard
            
        Returns:
            List of FileShard instances for the oversized item
        """
        self.logger.warning(f"Handling oversized item at index {index}")
        
        if isinstance(item, dict):
            # Use dict processor for nested dictionary
            from .dict_processor import DictProcessor
            dict_processor = DictProcessor(self.sharding_engine, self.size_calculator)
            result = dict_processor.process(item, max_size)
            
            # Update metadata to reflect list context
            for shard in result.shards:
                shard.metadata.original_path = [f"[{index}]"] + shard.metadata.original_path
            
            return result.shards
        
        elif isinstance(item, list):
            # Recursively process nested list
            nested_result = self.process(item, max_size)
            
            # Update metadata to reflect list context
            for shard in nested_result.shards:
                shard.metadata.original_path = [f"[{index}]"] + shard.metadata.original_path
            
            return nested_result.shards
        
        else:
            # Oversized primitive - create shard with warning
            shard_id = self.sharding_engine.generate_shard_id()
            
            metadata = ShardMetadata(
                shard_id=shard_id,
                parent_id=None,
                child_ids=[],
                data_type=DataType.PRIMITIVE,
                original_path=[f"[{index}]"]
            )
            
            # Add warning to the data
            shard_data = {
                "_warning": "Oversized primitive item",
                "_original_index": index,
                "_item": item
            }
            
            shard = self.sharding_engine.create_shard(shard_data, shard_id, metadata)
            return [shard]
    
    def _setup_sequential_linking(self, shards: List[FileShard]) -> None:
        """
        Set up sequential linking between shards.
        
        Args:
            shards: List of shards to link
        """
        for i in range(len(shards)):
            if i > 0:
                shards[i].metadata.previous_shard = shards[i-1].id
            if i < len(shards) - 1:
                shards[i].metadata.next_shard = shards[i+1].id
    
    def _create_empty_list_shard(self) -> FileShard:
        """
        Create a shard for an empty list.
        
        Returns:
            FileShard for empty list
        """
        shard_id = self.sharding_engine.generate_shard_id()
        
        metadata = ShardMetadata(
            shard_id=shard_id,
            parent_id=None,
            child_ids=[],
            data_type=DataType.LIST,
            original_path=["[]"]
        )
        
        return self.sharding_engine.create_shard([], shard_id, metadata)
    
    def optimize_list_processing(self, data: List[Any], max_size: int) -> dict:
        """
        Analyze list structure and provide optimization recommendations.
        
        Args:
            data: List data to analyze
            max_size: Target maximum size per shard
            
        Returns:
            Dictionary with optimization recommendations
        """
        if not data:
            return {
                "total_size": 0,
                "estimated_shards": 1,
                "item_count": 0,
                "optimization_suggestions": []
            }
        
        total_size = self.size_calculator.calculate_json_size(data)
        analysis = self._analyze_list_structure(data)
        
        # Calculate item size distribution
        item_sizes = []
        large_items = []
        
        for i, item in enumerate(data):
            try:
                item_size = self.size_calculator.calculate_json_size(item)
                item_sizes.append(item_size)
                
                if item_size > max_size * 0.5:  # Items larger than 50% of max size
                    large_items.append((i, item_size))
            except ValueError:
                continue
        
        # Estimate shard count
        estimated_shards = self.size_calculator.estimate_shard_count(total_size, max_size)
        
        recommendations = {
            "total_size": total_size,
            "estimated_shards": estimated_shards,
            "item_count": len(data),
            "analysis": analysis,
            "large_items": large_items,
            "average_item_size": sum(item_sizes) / len(item_sizes) if item_sizes else 0,
            "max_item_size": max(item_sizes) if item_sizes else 0,
            "optimization_suggestions": []
        }
        
        # Add optimization suggestions
        if large_items:
            recommendations["optimization_suggestions"].append({
                "type": "split_large_items",
                "description": f"Consider splitting {len(large_items)} large items",
                "affected_indices": [i for i, _ in large_items]
            })
        
        if not analysis["homogeneous"]:
            recommendations["optimization_suggestions"].append({
                "type": "group_similar_items",
                "description": "List contains mixed item types. Consider grouping similar items."
            })
        
        if analysis["has_nested_structures"]:
            recommendations["optimization_suggestions"].append({
                "type": "flatten_nested_structures",
                "description": "List contains nested structures. Consider flattening for better performance."
            })
        
        if len(data) > 10000:
            recommendations["optimization_suggestions"].append({
                "type": "batch_processing",
                "description": "Very large list. Consider batch processing strategies."
            })
        
        return recommendations
    
    def validate_list_ordering(self, shards: List[FileShard]) -> bool:
        """
        Validate that list shards maintain proper ordering.
        
        Args:
            shards: List of shards to validate
            
        Returns:
            True if ordering is correct
        """
        if not shards:
            return True
        
        # Check sequential linking
        for i, shard in enumerate(shards):
            if i > 0:
                if shard.metadata.previous_shard != shards[i-1].id:
                    self.logger.error(f"Broken previous link at shard {i}: "
                                    f"expected {shards[i-1].id}, got {shard.metadata.previous_shard}")
                    return False
            
            if i < len(shards) - 1:
                if shard.metadata.next_shard != shards[i+1].id:
                    self.logger.error(f"Broken next link at shard {i}: "
                                    f"expected {shards[i+1].id}, got {shard.metadata.next_shard}")
                    return False
        
        return True
    
    def get_list_statistics(self, shards: List[FileShard]) -> dict:
        """
        Get statistics about list processing results.
        
        Args:
            shards: List of processed shards
            
        Returns:
            Dictionary with statistics
        """
        if not shards:
            return {
                "total_shards": 0,
                "total_items": 0,
                "average_items_per_shard": 0,
                "sequential_integrity": True
            }
        
        total_items = 0
        for shard in shards:
            if isinstance(shard.data, list):
                total_items += len(shard.data)
        
        return {
            "total_shards": len(shards),
            "total_items": total_items,
            "average_items_per_shard": total_items / len(shards) if shards else 0,
            "sequential_integrity": self.validate_list_ordering(shards),
            "size_distribution": self.sharding_engine.get_sharding_statistics(shards)
        }