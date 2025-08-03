"""Size calculation utilities for JSON data and file operations."""

import json
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple
from ..types import DataType


class SizeCalculator:
    """
    Utility class for calculating sizes of JSON data and estimating file overhead.
    
    Provides accurate size calculations in UTF-8 bytes for JSON serialization,
    metadata overhead estimation, and buffer calculations for formatting.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the size calculator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Constants for size calculations
        self.METADATA_BASE_SIZE = 300  # Base metadata overhead in bytes (increased for 90KB chunks)
        self.FORMATTING_BUFFER_RATIO = 0.08  # 8% buffer for formatting (optimized for larger chunks)
        self.FILE_SYSTEM_OVERHEAD = 64  # Estimated filesystem overhead per file (4KB block aligned)
    
    def calculate_json_size(self, data: Any, ensure_ascii: bool = False, 
                           use_cache: bool = True) -> int:
        """
        Calculate the size of data when serialized to JSON in UTF-8 bytes.
        
        Args:
            data: Data to calculate size for
            ensure_ascii: Whether to ensure ASCII encoding
            use_cache: Whether to use size estimation cache for performance
            
        Returns:
            Size in bytes
            
        Raises:
            ValueError: If data is not JSON serializable
        """
        try:
            # Fast path for primitives
            if isinstance(data, (str, int, float, bool)) or data is None:
                return self._calculate_primitive_size(data, ensure_ascii)
            
            # Use optimized compact serialization
            json_string = json.dumps(data, ensure_ascii=ensure_ascii, separators=(',', ':'))
            return len(json_string.encode('utf-8'))
        except (TypeError, ValueError, RecursionError) as e:
            raise ValueError(f"Data is not JSON serializable: {str(e)}")
    
    def _calculate_primitive_size(self, data: Any, ensure_ascii: bool = False) -> int:
        """Fast size calculation for primitive types."""
        if data is None:
            return 4  # "null"
        elif isinstance(data, bool):
            return 5 if data else 4  # "true" or "false"
        elif isinstance(data, int):
            return len(str(data))
        elif isinstance(data, float):
            return len(json.dumps(data))
        elif isinstance(data, str):
            # Fast estimation for strings
            if ensure_ascii:
                return len(json.dumps(data, ensure_ascii=True))
            else:
                # Estimate: string length + quotes + escape overhead
                return len(data.encode('utf-8')) + 2 + (data.count('"') + data.count('\\'))
        return 0
    
    def calculate_formatted_json_size(self, data: Any, indent: int = 2) -> int:
        """
        Calculate the size of data when serialized to formatted JSON.
        
        Args:
            data: Data to calculate size for
            indent: Indentation level for formatting
            
        Returns:
            Size in bytes
        """
        try:
            json_string = json.dumps(data, ensure_ascii=False, indent=indent)
            return len(json_string.encode('utf-8'))
        except (TypeError, ValueError, RecursionError) as e:
            raise ValueError(f"Data is not JSON serializable: {str(e)}")
    
    def calculate_metadata_overhead(self, shard_id: str, parent_id: Optional[str],
                                  child_ids: List[str], data_type: DataType,
                                  original_path: List[str]) -> int:
        """
        Calculate the overhead size of shard metadata.
        
        Args:
            shard_id: Shard identifier
            parent_id: Parent shard identifier
            child_ids: List of child shard identifiers
            data_type: Type of data in the shard
            original_path: Original path in the data structure
            
        Returns:
            Metadata overhead size in bytes
        """
        metadata = {
            "shardId": shard_id,
            "parentId": parent_id,
            "childIds": child_ids,
            "dataType": data_type.value,
            "originalPath": original_path,
            "nextShard": None,
            "previousShard": None,
            "version": "1.0.0"
        }
        
        return self.calculate_json_size(metadata)
    
    def calculate_file_schema_size(self, data: Any, metadata_size: int) -> int:
        """
        Calculate the total size of a file schema including data and metadata.
        
        Args:
            data: Shard data
            metadata_size: Pre-calculated metadata size
            
        Returns:
            Total file schema size in bytes
        """
        data_size = self.calculate_json_size(data)
        
        # Account for the schema structure: {"_metadata": {...}, "data": {...}}
        schema_overhead = len('{"_metadata":,"data":}'.encode('utf-8'))
        
        return data_size + metadata_size + schema_overhead
    
    def calculate_total_file_size(self, data: Any, metadata_size: int, 
                                formatted: bool = True) -> int:
        """
        Calculate the total file size including all overhead and formatting.
        
        Args:
            data: Shard data
            metadata_size: Pre-calculated metadata size
            formatted: Whether the JSON will be formatted
            
        Returns:
            Total estimated file size in bytes
        """
        if formatted:
            base_size = self.calculate_file_schema_size_formatted(data, metadata_size)
        else:
            base_size = self.calculate_file_schema_size(data, metadata_size)
        
        # Add formatting buffer
        buffer_size = int(base_size * self.FORMATTING_BUFFER_RATIO)
        
        # Add filesystem overhead
        total_size = base_size + buffer_size + self.FILE_SYSTEM_OVERHEAD
        
        return total_size
    
    def calculate_file_schema_size_formatted(self, data: Any, metadata_size: int) -> int:
        """
        Calculate the size of a formatted file schema.
        
        Args:
            data: Shard data
            metadata_size: Pre-calculated metadata size (will be recalculated for formatting)
            
        Returns:
            Formatted file schema size in bytes
        """
        # Create a sample schema to calculate formatted size
        sample_metadata = {
            "shardId": "sample_001",
            "parentId": None,
            "childIds": [],
            "dataType": "dict",
            "originalPath": [],
            "nextShard": None,
            "previousShard": None,
            "version": "1.0.0"
        }
        
        sample_schema = {
            "_metadata": sample_metadata,
            "data": data
        }
        
        return self.calculate_formatted_json_size(sample_schema)
    
    def estimate_shard_count(self, total_size: int, max_shard_size: int, 
                           overhead_ratio: float = 0.2) -> int:
        """
        Estimate the number of shards needed for given total size.
        
        Args:
            total_size: Total data size in bytes
            max_shard_size: Maximum size per shard in bytes
            overhead_ratio: Ratio of overhead to account for (default 20%)
            
        Returns:
            Estimated number of shards needed
        """
        # Account for metadata and formatting overhead
        effective_shard_size = int(max_shard_size * (1 - overhead_ratio))
        
        # Calculate number of shards (ceiling division)
        shard_count = (total_size + effective_shard_size - 1) // effective_shard_size
        
        return max(1, shard_count)  # At least 1 shard
    
    def calculate_size_breakdown(self, data: Any, shard_id: str = "sample_001",
                               parent_id: Optional[str] = None,
                               child_ids: Optional[List[str]] = None,
                               data_type: DataType = DataType.DICT,
                               original_path: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Calculate a detailed size breakdown for a shard.
        
        Args:
            data: Shard data
            shard_id: Shard identifier
            parent_id: Parent shard identifier
            child_ids: List of child shard identifiers
            data_type: Type of data in the shard
            original_path: Original path in the data structure
            
        Returns:
            Dictionary with detailed size breakdown
        """
        if child_ids is None:
            child_ids = []
        if original_path is None:
            original_path = []
        
        # Calculate individual components
        data_size = self.calculate_json_size(data)
        metadata_size = self.calculate_metadata_overhead(
            shard_id, parent_id, child_ids, data_type, original_path
        )
        schema_size = self.calculate_file_schema_size(data, metadata_size)
        formatted_size = self.calculate_file_schema_size_formatted(data, metadata_size)
        total_size = self.calculate_total_file_size(data, metadata_size, formatted=True)
        
        return {
            "data_size": data_size,
            "metadata_size": metadata_size,
            "schema_overhead": schema_size - data_size - metadata_size,
            "schema_size": schema_size,
            "formatted_size": formatted_size,
            "formatting_buffer": int(formatted_size * self.FORMATTING_BUFFER_RATIO),
            "filesystem_overhead": self.FILE_SYSTEM_OVERHEAD,
            "total_size": total_size
        }
    
    def will_exceed_size_limit(self, current_data: Any, new_data: Any, 
                             max_size: int, metadata_overhead: int = None) -> bool:
        """
        Check if adding new data to current data will exceed size limit.
        
        Args:
            current_data: Current shard data
            new_data: New data to be added
            max_size: Maximum allowed size
            metadata_overhead: Pre-calculated metadata overhead (optional)
            
        Returns:
            True if size limit would be exceeded
        """
        try:
            # Calculate current size
            current_size = self.calculate_json_size(current_data)
            
            # Calculate new data size
            new_data_size = self.calculate_json_size(new_data)
            
            # Estimate metadata overhead if not provided
            if metadata_overhead is None:
                metadata_overhead = self.METADATA_BASE_SIZE
            
            # Calculate total estimated size
            total_size = current_size + new_data_size + metadata_overhead
            
            # Add formatting buffer
            total_size += int(total_size * self.FORMATTING_BUFFER_RATIO)
            
            # Add filesystem overhead
            total_size += self.FILE_SYSTEM_OVERHEAD
            
            return total_size > max_size
            
        except ValueError:
            # If size calculation fails, assume it would exceed limit
            return True
    
    def calculate_memory_usage(self, data: Any) -> Dict[str, int]:
        """
        Calculate approximate memory usage of data structure.
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary with memory usage information
        """
        # Get size of the object in memory
        memory_size = sys.getsizeof(data)
        
        # Calculate JSON serialized size
        try:
            json_size = self.calculate_json_size(data)
        except ValueError:
            json_size = -1
        
        # Estimate deep memory usage for nested structures
        deep_memory = self._calculate_deep_memory_usage(data)
        
        return {
            "shallow_memory": memory_size,
            "deep_memory": deep_memory,
            "json_size": json_size,
            "memory_to_json_ratio": deep_memory / json_size if json_size > 0 else -1
        }
    
    def _calculate_deep_memory_usage(self, data: Any, visited: Optional[set] = None) -> int:
        """
        Calculate deep memory usage including nested structures.
        
        Args:
            data: Data to analyze
            visited: Set of visited object IDs to avoid circular references
            
        Returns:
            Deep memory usage in bytes
        """
        if visited is None:
            visited = set()
        
        # Avoid circular references
        obj_id = id(data)
        if obj_id in visited:
            return 0
        visited.add(obj_id)
        
        total_size = sys.getsizeof(data)
        
        try:
            if isinstance(data, dict):
                for key, value in data.items():
                    total_size += sys.getsizeof(key)
                    total_size += self._calculate_deep_memory_usage(value, visited)
            elif isinstance(data, (list, tuple)):
                for item in data:
                    total_size += self._calculate_deep_memory_usage(item, visited)
            elif isinstance(data, str):
                # String size is already included in sys.getsizeof
                pass
        finally:
            visited.remove(obj_id)
        
        return total_size
    
    def optimize_for_size_limit(self, data: Any, max_size: int) -> Dict[str, Any]:
        """
        Provide optimization suggestions for fitting data within size limit.
        
        Args:
            data: Data to optimize
            max_size: Target maximum size
            
        Returns:
            Dictionary with optimization suggestions
        """
        current_size = self.calculate_total_file_size(data, self.METADATA_BASE_SIZE)
        
        if current_size <= max_size:
            return {
                "needs_optimization": False,
                "current_size": current_size,
                "target_size": max_size,
                "suggestions": []
            }
        
        suggestions = []
        size_reduction_needed = current_size - max_size
        
        # Analyze data structure for optimization opportunities
        if isinstance(data, dict):
            # Suggest splitting large dictionary values
            large_values = []
            for key, value in data.items():
                value_size = self.calculate_json_size(value)
                if value_size > max_size * 0.3:  # Values larger than 30% of max size
                    large_values.append((key, value_size))
            
            if large_values:
                suggestions.append({
                    "type": "split_large_values",
                    "description": f"Split {len(large_values)} large dictionary values",
                    "potential_reduction": sum(size for _, size in large_values) * 0.5
                })
        
        elif isinstance(data, list):
            # Suggest splitting large list
            if len(data) > 10:
                suggestions.append({
                    "type": "split_list",
                    "description": f"Split list of {len(data)} items into smaller chunks",
                    "potential_reduction": current_size * 0.6
                })
        
        # Suggest removing formatting if size is close
        if size_reduction_needed < current_size * 0.1:  # Less than 10% reduction needed
            suggestions.append({
                "type": "remove_formatting",
                "description": "Use compact JSON formatting",
                "potential_reduction": current_size * self.FORMATTING_BUFFER_RATIO
            })
        
        return {
            "needs_optimization": True,
            "current_size": current_size,
            "target_size": max_size,
            "size_reduction_needed": size_reduction_needed,
            "suggestions": suggestions
        }