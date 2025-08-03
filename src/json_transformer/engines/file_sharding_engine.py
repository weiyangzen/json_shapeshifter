"""File sharding engine for splitting JSON data into manageable chunks."""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple
from ..types import FileShardingEngineInterface, DataType
from ..models import FileShard, ShardMetadata
from ..utils.size_calculator import SizeCalculator


class FileShardingEngine(FileShardingEngineInterface):
    """
    File sharding engine that splits JSON data into size-constrained shards.
    
    Manages the creation of file shards with proper metadata linking
    and size constraint enforcement.
    """
    
    def __init__(self, size_calculator: Optional[SizeCalculator] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the file sharding engine.
        
        Args:
            size_calculator: Optional SizeCalculator instance
            logger: Optional logger instance
        """
        self.size_calculator = size_calculator or SizeCalculator()
        self.logger = logger or logging.getLogger(__name__)
        self._shard_counter = 0
    
    def create_shard(self, data: Any, shard_id: str, metadata: ShardMetadata) -> FileShard:
        """
        Create a new file shard with data and metadata.
        
        Args:
            data: Data to include in the shard
            shard_id: Unique identifier for the shard
            metadata: Shard metadata
            
        Returns:
            FileShard instance
            
        Raises:
            ValueError: If shard creation fails
        """
        try:
            # Calculate shard size
            shard_size = self.calculate_size(data)
            
            # Validate that shard_id matches metadata
            if shard_id != metadata.shard_id:
                raise ValueError(f"Shard ID mismatch: {shard_id} != {metadata.shard_id}")
            
            # Create the shard
            shard = FileShard(
                id=shard_id,
                data=data,
                metadata=metadata,
                size=shard_size
            )
            
            self.logger.debug(f"Created shard {shard_id} with size {shard_size} bytes")
            return shard
            
        except Exception as e:
            self.logger.error(f"Failed to create shard {shard_id}: {e}")
            raise ValueError(f"Shard creation failed: {str(e)}")
    
    def calculate_size(self, data: Any) -> int:
        """
        Calculate the size of data when serialized as a shard.
        
        Args:
            data: Data to calculate size for
            
        Returns:
            Size in bytes
        """
        try:
            # Calculate the size including metadata overhead
            metadata_size = self.size_calculator.METADATA_BASE_SIZE
            return self.size_calculator.calculate_total_file_size(
                data, metadata_size, formatted=True
            )
        except ValueError as e:
            self.logger.warning(f"Size calculation failed: {e}")
            return -1  # Indicate calculation failure
    
    def should_create_new_shard(self, current_size: int, new_data_size: int, 
                              max_size: int) -> bool:
        """
        Determine if a new shard should be created based on size constraints.
        Uses advanced heuristics for optimal shard sizing.
        
        Args:
            current_size: Current shard size in bytes
            new_data_size: Size of new data to be added
            max_size: Maximum allowed shard size
            
        Returns:
            True if a new shard should be created
        """
        # Adaptive overhead calculation based on shard size
        if max_size >= 90000:  # 90KB+
            overhead_ratio = 0.08  # 8% overhead for large shards
        elif max_size >= 50000:  # 50KB+
            overhead_ratio = 0.10  # 10% overhead for medium shards
        else:
            overhead_ratio = 0.15  # 15% overhead for small shards
        
        overhead_buffer = int(max_size * overhead_ratio)
        effective_max_size = max_size - overhead_buffer
        
        projected_size = current_size + new_data_size
        
        # Advanced heuristic: prefer fuller shards for better compression
        utilization_threshold = 0.85  # Target 85% utilization
        target_size = int(effective_max_size * utilization_threshold)
        
        # Create new shard if we exceed target or would exceed max
        should_create = (projected_size > target_size and current_size > 0) or \
                       (projected_size > effective_max_size)
        
        if should_create:
            self.logger.debug(f"New shard needed: projected size {projected_size} > "
                            f"target {target_size} (effective max {effective_max_size})")
        
        return should_create
    
    def generate_shard_id(self, prefix: str = "shard") -> str:
        """
        Generate a unique shard identifier.
        
        Args:
            prefix: Prefix for the shard ID
            
        Returns:
            Unique shard identifier
        """
        self._shard_counter += 1
        return f"{prefix}_{self._shard_counter:03d}"
    
    def split_data_by_size(self, data: Any, max_size: int, 
                          base_path: List[str] = None) -> List[Tuple[Any, List[str]]]:
        """
        Split data into chunks that fit within size constraints.
        
        Args:
            data: Data to split
            max_size: Maximum size per chunk
            base_path: Base path for the data
            
        Returns:
            List of (chunk_data, path) tuples
        """
        if base_path is None:
            base_path = []
        
        chunks = []
        
        if isinstance(data, dict):
            chunks.extend(self._split_dict_by_size(data, max_size, base_path))
        elif isinstance(data, list):
            chunks.extend(self._split_list_by_size(data, max_size, base_path))
        else:
            # Primitive data - check if it fits
            data_size = self.calculate_size(data)
            if data_size <= max_size:
                chunks.append((data, base_path))
            else:
                self.logger.warning(f"Primitive data at {base_path} exceeds max size")
                # Still include it - will be handled by caller
                chunks.append((data, base_path))
        
        return chunks
    
    def _split_dict_by_size(self, data: Dict[str, Any], max_size: int, 
                           base_path: List[str]) -> List[Tuple[Any, List[str]]]:
        """Split dictionary data by size constraints."""
        chunks = []
        current_chunk = {}
        current_size = 0
        
        for key, value in data.items():
            current_path = base_path + [key]
            
            # Calculate size of this key-value pair
            pair_data = {key: value}
            pair_size = self.calculate_size(pair_data)
            
            # If this single pair exceeds max size, handle it separately
            if pair_size > max_size:
                # Save current chunk if it has data
                if current_chunk:
                    chunks.append((current_chunk, base_path))
                    current_chunk = {}
                    current_size = 0
                
                # Handle oversized value
                if isinstance(value, (dict, list)):
                    # Recursively split the oversized value
                    sub_chunks = self.split_data_by_size(value, max_size, current_path)
                    chunks.extend(sub_chunks)
                else:
                    # Oversized primitive - include as is with warning
                    self.logger.warning(f"Oversized primitive at {current_path}")
                    chunks.append((value, current_path))
                
                continue
            
            # Check if adding this pair would exceed size limit
            if self.should_create_new_shard(current_size, pair_size, max_size):
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append((current_chunk, base_path))
                
                current_chunk = {key: value}
                current_size = pair_size
            else:
                # Add to current chunk
                current_chunk[key] = value
                current_size += pair_size
        
        # Add final chunk if it has data
        if current_chunk:
            chunks.append((current_chunk, base_path))
        
        return chunks
    
    def _split_list_by_size(self, data: List[Any], max_size: int, 
                           base_path: List[str]) -> List[Tuple[Any, List[str]]]:
        """Split list data by size constraints."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, item in enumerate(data):
            current_path = base_path + [f"[{i}]"]
            
            # Calculate size of this item
            item_size = self.calculate_size(item)
            
            # If this single item exceeds max size, handle it separately
            if item_size > max_size:
                # Save current chunk if it has data
                if current_chunk:
                    chunks.append((current_chunk, base_path))
                    current_chunk = []
                    current_size = 0
                
                # Handle oversized item
                if isinstance(item, (dict, list)):
                    # Recursively split the oversized item
                    sub_chunks = self.split_data_by_size(item, max_size, current_path)
                    chunks.extend(sub_chunks)
                else:
                    # Oversized primitive - include as is with warning
                    self.logger.warning(f"Oversized primitive at {current_path}")
                    chunks.append((item, current_path))
                
                continue
            
            # Check if adding this item would exceed size limit
            if self.should_create_new_shard(current_size, item_size, max_size):
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append((current_chunk, base_path))
                
                current_chunk = [item]
                current_size = item_size
            else:
                # Add to current chunk
                current_chunk.append(item)
                current_size += item_size
        
        # Add final chunk if it has data
        if current_chunk:
            chunks.append((current_chunk, base_path))
        
        return chunks
    
    def create_shards_from_chunks(self, chunks: List[Tuple[Any, List[str]]], 
                                 parent_id: Optional[str] = None) -> List[FileShard]:
        """
        Create file shards from data chunks.
        
        Args:
            chunks: List of (data, path) tuples
            parent_id: Optional parent shard ID
            
        Returns:
            List of FileShard instances
        """
        shards = []
        
        for i, (chunk_data, chunk_path) in enumerate(chunks):
            # Generate shard ID
            shard_id = self.generate_shard_id()
            
            # Determine data type
            if isinstance(chunk_data, dict):
                data_type = DataType.DICT
            elif isinstance(chunk_data, list):
                data_type = DataType.LIST
            else:
                data_type = DataType.PRIMITIVE
            
            # Create metadata
            metadata = ShardMetadata(
                shard_id=shard_id,
                parent_id=parent_id,
                child_ids=[],  # Will be populated later if needed
                data_type=data_type,
                original_path=chunk_path
            )
            
            # Set up sequential linking for multiple chunks
            if i > 0:
                metadata.previous_shard = shards[i-1].id
                shards[i-1].metadata.next_shard = shard_id
            
            # Create shard
            shard = self.create_shard(chunk_data, shard_id, metadata)
            shards.append(shard)
        
        return shards
    
    def optimize_shard_distribution(self, shards: List[FileShard], 
                                   max_size: int) -> List[FileShard]:
        """
        Optimize the distribution of data across shards.
        
        Args:
            shards: List of shards to optimize
            max_size: Maximum size per shard
            
        Returns:
            Optimized list of shards
        """
        if not shards:
            return shards
        
        optimized_shards = []
        
        for shard in shards:
            if shard.size <= max_size:
                # Shard is within size limit
                optimized_shards.append(shard)
            else:
                # Shard exceeds size limit - needs further splitting
                self.logger.info(f"Optimizing oversized shard {shard.id} "
                               f"({shard.size} > {max_size})")
                
                # Split the oversized shard
                chunks = self.split_data_by_size(
                    shard.data, max_size, shard.metadata.original_path
                )
                
                # Create new shards from chunks
                new_shards = self.create_shards_from_chunks(
                    chunks, shard.metadata.parent_id
                )
                
                # Update metadata to maintain relationships
                for new_shard in new_shards:
                    new_shard.metadata.original_path = shard.metadata.original_path
                
                optimized_shards.extend(new_shards)
        
        return optimized_shards
    
    def get_sharding_statistics(self, shards: List[FileShard]) -> Dict[str, Any]:
        """
        Get statistics about the sharding results.
        
        Args:
            shards: List of shards to analyze
            
        Returns:
            Dictionary with sharding statistics
        """
        if not shards:
            return {
                "total_shards": 0,
                "total_size": 0,
                "average_size": 0,
                "size_distribution": {},
                "data_type_distribution": {}
            }
        
        total_size = sum(shard.size for shard in shards)
        sizes = [shard.size for shard in shards]
        
        # Size distribution
        size_ranges = {
            "small (< 5KB)": sum(1 for s in sizes if s < 5000),
            "medium (5-15KB)": sum(1 for s in sizes if 5000 <= s < 15000),
            "large (15-25KB)": sum(1 for s in sizes if 15000 <= s < 25000),
            "oversized (> 25KB)": sum(1 for s in sizes if s >= 25000)
        }
        
        # Data type distribution
        data_types = [shard.metadata.data_type.value for shard in shards]
        type_counts = {}
        for dt in data_types:
            type_counts[dt] = type_counts.get(dt, 0) + 1
        
        return {
            "total_shards": len(shards),
            "total_size": total_size,
            "average_size": total_size / len(shards),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "size_distribution": size_ranges,
            "data_type_distribution": type_counts,
            "efficiency": self._calculate_efficiency(shards, 25000)  # Assume 25KB default
        }
    
    def _calculate_efficiency(self, shards: List[FileShard], target_size: int) -> float:
        """
        Calculate sharding efficiency (how well shards utilize the target size).
        
        Args:
            shards: List of shards
            target_size: Target size per shard
            
        Returns:
            Efficiency ratio (0.0 to 1.0)
        """
        if not shards:
            return 0.0
        
        total_actual_size = sum(shard.size for shard in shards)
        total_possible_size = len(shards) * target_size
        
        return total_actual_size / total_possible_size if total_possible_size > 0 else 0.0