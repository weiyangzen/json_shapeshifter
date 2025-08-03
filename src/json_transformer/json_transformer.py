"""Main JSON Transformer implementation with state-of-the-art optimizations."""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from .types import (
    JSONTransformerInterface,
    UnflattenResult,
    FlattenResult,
    DataStructure,
    ProcessingError,
    ErrorType
)
from .parser import JSONParser
from .data_type_detector import DataTypeDetector
from .processors import DictProcessor, ListProcessor, MixedProcessor
from .io.file_writer import FileWriter
from .error_handler import ErrorHandler
from .profiler import PerformanceProfiler
from .streaming import StreamingProcessor, CompressionOptimizer


class JSONTransformer(JSONTransformerInterface):
    """
    Main implementation of the JSON Transformer interface.
    
    Provides bidirectional conversion between compact JSON format
    and distributed LLM-readable file structures.
    """
    
    def __init__(self, default_max_size: int = 92160, 
                 logger: Optional[logging.Logger] = None,
                 enable_parallel_processing: bool = True,
                 max_workers: Optional[int] = None,
                 enable_compression: bool = True,
                 memory_limit_mb: int = 1024):
        """
        Initialize the JSON Transformer with state-of-the-art optimizations.
        
        Args:
            default_max_size: Default maximum file size in bytes (90KB)
            logger: Optional logger instance
            enable_parallel_processing: Enable parallel processing for large datasets
            max_workers: Maximum number of worker threads (None = auto-detect)
            enable_compression: Enable compression optimizations
            memory_limit_mb: Memory limit in MB for processing
        """
        self.default_max_size = default_max_size
        self.logger = logger or logging.getLogger(__name__)
        self.enable_parallel_processing = enable_parallel_processing
        self.max_workers = max_workers
        self.enable_compression = enable_compression
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        
        # Initialize thread pool for parallel processing
        if enable_parallel_processing:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self.executor = None
        
        # Initialize components with optimizations
        self.error_handler = ErrorHandler(self.logger)
        self.parser = JSONParser(self.error_handler, self.logger)
        self.detector = DataTypeDetector(self.logger)
        self.dict_processor = DictProcessor(
            logger=self.logger,
            enable_streaming=True,
            compression_threshold=self.default_max_size // 2
        )
        self.list_processor = ListProcessor(
            logger=self.logger,
            enable_streaming=True,
            batch_size=min(1000, self.default_max_size // 100)
        )
        self.mixed_processor = MixedProcessor(logger=self.logger)
        self.file_writer = FileWriter(self.logger)
        
        # Advanced components
        self.profiler = PerformanceProfiler(self.logger)
        self.streaming_processor = StreamingProcessor(
            chunk_size=default_max_size,
            logger=self.logger
        )
        self.compression_optimizer = CompressionOptimizer(self.logger) if enable_compression else None
    
    async def unflatten(
        self, 
        json_string: str, 
        max_size: Optional[int] = None, 
        output_dir: Optional[str] = None
    ) -> UnflattenResult:
        """
        Unflatten JSON string into multiple LLM-readable files.
        
        Args:
            json_string: Input JSON string to unflatten
            max_size: Maximum file size in bytes (defaults to 90KB)
            output_dir: Output directory path (defaults to './output')
            
        Returns:
            UnflattenResult with operation details
        """
        try:
            # Set defaults
            if max_size is None:
                max_size = self.default_max_size
            if output_dir is None:
                output_dir = "./output"
            
            # Calculate input size and start profiling
            input_size = len(json_string.encode('utf-8'))
            
            with self.profiler.profile_operation("unflatten_json", input_size):
                self.logger.info(f"Starting unflatten operation: max_size={max_size}B ({max_size/1024:.1f}KB), output_dir={output_dir}")
                
                # Advanced memory usage estimation
                memory_estimate = self.streaming_processor.estimate_memory_usage(json_string)
                
                if memory_estimate["total_estimated"] > self.memory_limit_bytes:
                    self.logger.warning(f"Large dataset detected ({memory_estimate['recommended_memory_mb']}MB recommended). "
                                      f"Consider processing in chunks or increasing memory limit.")
                
                self.logger.info(f"Input size: {input_size/1024:.1f}KB, estimated memory: {memory_estimate['recommended_memory_mb']}MB")
                
                # Sample performance during processing
                self.profiler.sample_performance()
            
            # Validate inputs
            size_validation = self.error_handler.validate_size_parameter(max_size)
            if not size_validation.is_valid:
                return UnflattenResult(
                    success=False,
                    output_directory=output_dir,
                    file_count=0,
                    total_size=0,
                    errors=[error.message for error in size_validation.errors]
                )
            
            dir_validation = self.error_handler.validate_directory_path(output_dir)
            if not dir_validation.is_valid:
                return UnflattenResult(
                    success=False,
                    output_directory=output_dir,
                    file_count=0,
                    total_size=0,
                    errors=[error.message for error in dir_validation.errors]
                )
            
                # Parse JSON
                try:
                    data, structure_type = self.parser.parse(json_string)
                    self.profiler.sample_performance()
                except ValueError as e:
                    return UnflattenResult(
                        success=False,
                        output_directory=output_dir,
                        file_count=0,
                        total_size=0,
                        errors=[str(e)]
                    )
            
            # Validate data for processing
            processing_validation = self.parser.validate_for_processing(data, max_size)
            if not processing_validation.is_valid:
                return UnflattenResult(
                    success=False,
                    output_directory=output_dir,
                    file_count=0,
                    total_size=0,
                    errors=[error.message for error in processing_validation.errors]
                )
            
            # Select appropriate processor
            if structure_type == DataStructure.DICT_OF_DICTS:
                processor = self.dict_processor
            elif structure_type == DataStructure.LIST:
                processor = self.list_processor
            else:  # MIXED
                processor = self.mixed_processor
            
                # Apply compression optimization if enabled
                if self.compression_optimizer:
                    self.logger.debug("Applying compression optimizations...")
                    data = self.compression_optimizer.optimize_json_structure(data)
                    self.profiler.sample_performance()
                
                # Process data into shards
                try:
                    processing_result = processor.process(data, max_size)
                    self.profiler.sample_performance()
                except Exception as e:
                    return UnflattenResult(
                        success=False,
                        output_directory=output_dir,
                        file_count=0,
                        total_size=0,
                        errors=[f"Processing failed: {str(e)}"]
                    )
            
            # Write shards to files
            try:
                write_result = self.file_writer.write_shards(
                    processing_result.shards,
                    output_dir,
                    processing_result.metadata
                )
                
                # Create index file
                index_result = self.file_writer.create_index_file(
                    output_dir,
                    processing_result.shards,
                    processing_result.metadata
                )
                
                if write_result["success"]:
                    # Complete profiling with output metrics
                    metrics = self.profiler.stop_profiling(
                        output_size=write_result["total_size"],
                        shards_created=len(processing_result.shards)
                    )
                    
                    # Advanced performance analysis
                    avg_shard_size = write_result["total_size"] / len(processing_result.shards)
                    efficiency_score = min(1.0, (avg_shard_size / max_size) * 1.2)  # Target 80% utilization
                    
                    self.logger.info(f"Advanced Metrics:")
                    self.logger.info(f"  Efficiency Score: {efficiency_score:.2f}")
                    self.logger.info(f"  Memory Efficiency: {input_size / (metrics.memory_peak_mb * 1024 * 1024):.2f}")
                    
                    # Provide optimization recommendations
                    if hasattr(self, 'profiler') and len(self.profiler.metrics_history) > 1:
                        recommendations = self.profiler.optimize_based_on_history()
                        if recommendations["recommendations"]:
                            self.logger.info("Optimization Recommendations:")
                            for rec in recommendations["recommendations"][:3]:  # Top 3
                                self.logger.info(f"  • {rec}")
                    
                    return UnflattenResult(
                        success=True,
                        output_directory=write_result["output_directory"],
                        file_count=len(processing_result.shards),
                        total_size=write_result["total_size"],
                        errors=write_result["errors"] if write_result["errors"] else None
                    )
                else:
                    return UnflattenResult(
                        success=False,
                        output_directory=output_dir,
                        file_count=len(write_result["files_written"]),
                        total_size=write_result["total_size"],
                        errors=write_result["errors"]
                    )
                    
            except ProcessingError as e:
                return UnflattenResult(
                    success=False,
                    output_directory=output_dir,
                    file_count=0,
                    total_size=0,
                    errors=[str(e)]
                )
        
        except Exception as e:
            self.logger.error(f"Unexpected error in unflatten: {e}")
            return UnflattenResult(
                success=False,
                output_directory=output_dir or "./output",
                file_count=0,
                total_size=0,
                errors=[f"Unexpected error: {str(e)}"]
            )
    
    async def flatten(self, input_dir: str) -> FlattenResult:
        """
        Flatten directory of files back into single JSON string.
        
        Args:
            input_dir: Directory containing LLM-readable files
            
        Returns:
            FlattenResult with reconstructed JSON
        """
        # Implementation will be added in task 9.2
        return FlattenResult(
            success=False,
            json_string="",
            errors=["Flatten functionality not yet implemented"]
        )