"""Advanced streaming capabilities for processing large datasets."""

import asyncio
import json
import logging
from typing import AsyncIterator, Any, Dict, List, Optional, Tuple
from .types import DataType
from .utils.size_calculator import SizeCalculator


class StreamingProcessor:
    """
    Advanced streaming processor for handling very large JSON datasets.
    
    Provides memory-efficient processing by streaming data in chunks
    without loading the entire dataset into memory.
    """
    
    def __init__(self, chunk_size: int = 92160, 
                 buffer_size: int = 1024 * 1024,  # 1MB buffer
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the streaming processor.
        
        Args:
            chunk_size: Target size for each chunk in bytes
            buffer_size: Internal buffer size for streaming
            logger: Optional logger instance
        """
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.logger = logger or logging.getLogger(__name__)
        self.size_calculator = SizeCalculator()
    
    async def stream_large_dict(self, data: Dict[str, Any]) -> AsyncIterator[Tuple[str, Any, int]]:
        """
        Stream large dictionary data in optimal chunks.
        
        Args:
            data: Dictionary to stream
            
        Yields:
            Tuples of (key, value, estimated_size)
        """
        current_chunk = {}
        current_size = 0
        
        for key, value in data.items():
            try:
                value_size = self.size_calculator.calculate_json_size({key: value})
                
                # If single item is too large, yield it separately
                if value_size > self.chunk_size:
                    if current_chunk:
                        yield ("_chunk", current_chunk, current_size)
                        current_chunk = {}
                        current_size = 0
                    
                    yield (key, value, value_size)
                    continue
                
                # Check if adding this item would exceed chunk size
                if current_size + value_size > self.chunk_size and current_chunk:
                    yield ("_chunk", current_chunk, current_size)
                    current_chunk = {}
                    current_size = 0
                
                current_chunk[key] = value
                current_size += value_size
                
                # Yield control periodically for async processing
                if len(current_chunk) % 100 == 0:
                    await asyncio.sleep(0)
                    
            except Exception as e:
                self.logger.warning(f"Skipping key '{key}' due to error: {e}")
                continue
        
        # Yield final chunk
        if current_chunk:
            yield ("_chunk", current_chunk, current_size)
    
    async def stream_large_list(self, data: List[Any]) -> AsyncIterator[Tuple[List[Any], int]]:
        """
        Stream large list data in optimal chunks.
        
        Args:
            data: List to stream
            
        Yields:
            Tuples of (chunk_data, estimated_size)
        """
        current_chunk = []
        current_size = 0
        
        for i, item in enumerate(data):
            try:
                item_size = self.size_calculator.calculate_json_size(item)
                
                # If single item is too large, yield it separately
                if item_size > self.chunk_size:
                    if current_chunk:
                        yield (current_chunk, current_size)
                        current_chunk = []
                        current_size = 0
                    
                    yield ([item], item_size)
                    continue
                
                # Check if adding this item would exceed chunk size
                if current_size + item_size > self.chunk_size and current_chunk:
                    yield (current_chunk, current_size)
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(item)
                current_size += item_size
                
                # Yield control periodically for async processing
                if i % 1000 == 0:
                    await asyncio.sleep(0)
                    
            except Exception as e:
                self.logger.warning(f"Skipping item at index {i} due to error: {e}")
                continue
        
        # Yield final chunk
        if current_chunk:
            yield (current_chunk, current_size)
    
    def estimate_memory_usage(self, data: Any) -> Dict[str, int]:
        """
        Estimate memory usage for processing data.
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary with memory usage estimates
        """
        try:
            json_size = self.size_calculator.calculate_json_size(data)
            
            # Conservative memory estimates
            parsing_overhead = json_size * 2  # JSON parsing overhead
            processing_overhead = json_size * 1.5  # Processing overhead
            output_overhead = json_size * 0.5  # Output generation overhead
            
            total_estimated = json_size + parsing_overhead + processing_overhead + output_overhead
            
            return {
                "input_size": json_size,
                "parsing_overhead": parsing_overhead,
                "processing_overhead": processing_overhead,
                "output_overhead": output_overhead,
                "total_estimated": total_estimated,
                "recommended_memory_mb": int(total_estimated / 1024 / 1024 * 1.2)  # 20% safety margin
            }
            
        except Exception as e:
            self.logger.error(f"Memory estimation failed: {e}")
            return {
                "input_size": -1,
                "total_estimated": -1,
                "recommended_memory_mb": 1024  # Default 1GB recommendation
            }
    
    async def adaptive_chunk_size(self, data: Any, target_memory_mb: int = 512) -> int:
        """
        Adaptively determine optimal chunk size based on data characteristics.
        
        Args:
            data: Sample data to analyze
            target_memory_mb: Target memory usage in MB
            
        Returns:
            Optimal chunk size in bytes
        """
        try:
            # Analyze sample data
            if isinstance(data, dict) and data:
                sample_key = next(iter(data.keys()))
                sample_size = self.size_calculator.calculate_json_size({sample_key: data[sample_key]})
                estimated_items = len(data)
            elif isinstance(data, list) and data:
                sample_size = self.size_calculator.calculate_json_size(data[0])
                estimated_items = len(data)
            else:
                return self.chunk_size  # Use default
            
            # Calculate optimal chunk size
            target_memory_bytes = target_memory_mb * 1024 * 1024
            items_per_chunk = max(1, target_memory_bytes // (sample_size * 3))  # 3x overhead
            optimal_chunk_size = min(self.chunk_size * 2, items_per_chunk * sample_size)
            
            # Ensure minimum viable chunk size
            optimal_chunk_size = max(self.chunk_size // 2, optimal_chunk_size)
            
            self.logger.info(f"Adaptive chunk size: {optimal_chunk_size} bytes "
                           f"(~{items_per_chunk} items per chunk)")
            
            return optimal_chunk_size
            
        except Exception as e:
            self.logger.warning(f"Adaptive chunk sizing failed: {e}, using default")
            return self.chunk_size


class CompressionOptimizer:
    """
    Advanced compression optimizer for reducing shard sizes.
    
    Provides intelligent compression strategies based on data patterns.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the compression optimizer."""
        self.logger = logger or logging.getLogger(__name__)
    
    def optimize_json_structure(self, data: Any) -> Any:
        """
        Optimize JSON structure for better compression and LLM processing.
        
        Args:
            data: Data to optimize
            
        Returns:
            Optimized data structure
        """
        if isinstance(data, dict):
            return self._optimize_dict(data)
        elif isinstance(data, list):
            return self._optimize_list(data)
        else:
            return data
    
    def _optimize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize dictionary structure."""
        optimized = {}
        
        # Group similar keys together for better compression
        sorted_keys = sorted(data.keys())
        
        for key in sorted_keys:
            value = data[key]
            if isinstance(value, (dict, list)):
                optimized[key] = self.optimize_json_structure(value)
            else:
                optimized[key] = value
        
        return optimized
    
    def _optimize_list(self, data: List[Any]) -> List[Any]:
        """Optimize list structure."""
        # For lists, maintain order but optimize nested structures
        return [
            self.optimize_json_structure(item) if isinstance(item, (dict, list)) else item
            for item in data
        ]
    
    def calculate_compression_potential(self, data: Any) -> Dict[str, float]:
        """
        Calculate potential compression ratios for different strategies.
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary with compression potential metrics
        """
        try:
            original_size = len(json.dumps(data, separators=(',', ':')))
            optimized_data = self.optimize_json_structure(data)
            optimized_size = len(json.dumps(optimized_data, separators=(',', ':')))
            
            # Estimate compression ratios for different algorithms
            # These are rough estimates based on typical JSON compression ratios
            gzip_ratio = 0.3  # Typical gzip compression for JSON
            lz4_ratio = 0.5   # Faster but less compression
            zstd_ratio = 0.25 # Better compression than gzip
            
            return {
                "original_size": original_size,
                "optimized_size": optimized_size,
                "structure_optimization_ratio": optimized_size / original_size,
                "estimated_gzip_ratio": gzip_ratio,
                "estimated_lz4_ratio": lz4_ratio,
                "estimated_zstd_ratio": zstd_ratio,
                "recommended_algorithm": "zstd" if original_size > 50000 else "lz4"
            }
            
        except Exception as e:
            self.logger.error(f"Compression analysis failed: {e}")
            return {
                "original_size": -1,
                "recommended_algorithm": "none"
            }