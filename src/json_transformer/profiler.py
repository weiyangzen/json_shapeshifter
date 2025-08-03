"""Advanced performance profiler for JSON Transformer operations."""

import time
import psutil
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class PerformanceMetrics:
    """Performance metrics for transformation operations."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    input_size: int
    output_size: int
    memory_peak_mb: float
    memory_start_mb: float
    memory_end_mb: float
    cpu_percent: float
    throughput_mbps: float
    shards_created: int
    compression_ratio: float


class PerformanceProfiler:
    """
    Advanced performance profiler for monitoring and optimizing operations.
    
    Provides detailed metrics on memory usage, CPU utilization, throughput,
    and other performance characteristics.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the performance profiler.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.metrics_history: List[PerformanceMetrics] = []
        self.current_operation: Optional[str] = None
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.peak_memory: float = 0
        self.cpu_samples: List[float] = []
    
    @contextmanager
    def profile_operation(self, operation_name: str, input_size: int = 0):
        """
        Context manager for profiling operations.
        
        Args:
            operation_name: Name of the operation being profiled
            input_size: Size of input data in bytes
        """
        self.start_profiling(operation_name, input_size)
        try:
            yield self
        finally:
            self.stop_profiling()
    
    def start_profiling(self, operation_name: str, input_size: int = 0):
        """
        Start profiling an operation.
        
        Args:
            operation_name: Name of the operation
            input_size: Size of input data in bytes
        """
        self.current_operation = operation_name
        self.start_time = time.time()
        self.input_size = input_size
        
        # Get initial memory usage
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        self.cpu_samples = []
        
        self.logger.debug(f"Started profiling: {operation_name}")
    
    def sample_performance(self):
        """Sample current performance metrics."""
        if not self.current_operation:
            return
        
        try:
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent = process.cpu_percent()
            
            self.peak_memory = max(self.peak_memory, current_memory)
            self.cpu_samples.append(cpu_percent)
            
        except Exception as e:
            self.logger.warning(f"Performance sampling failed: {e}")
    
    def stop_profiling(self, output_size: int = 0, shards_created: int = 0) -> PerformanceMetrics:
        """
        Stop profiling and return metrics.
        
        Args:
            output_size: Size of output data in bytes
            shards_created: Number of shards created
            
        Returns:
            PerformanceMetrics object with collected data
        """
        if not self.current_operation or not self.start_time:
            raise ValueError("No active profiling session")
        
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Get final memory usage
        try:
            process = psutil.Process()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        except Exception:
            end_memory = self.start_memory
            avg_cpu = 0
        
        # Calculate metrics
        throughput = (self.input_size / 1024 / 1024) / duration if duration > 0 else 0  # MB/s
        compression_ratio = output_size / self.input_size if self.input_size > 0 else 1.0
        
        metrics = PerformanceMetrics(
            operation_name=self.current_operation,
            start_time=self.start_time,
            end_time=end_time,
            duration=duration,
            input_size=self.input_size,
            output_size=output_size,
            memory_peak_mb=self.peak_memory,
            memory_start_mb=self.start_memory,
            memory_end_mb=end_memory,
            cpu_percent=avg_cpu,
            throughput_mbps=throughput,
            shards_created=shards_created,
            compression_ratio=compression_ratio
        )
        
        self.metrics_history.append(metrics)
        
        # Log performance summary
        self.logger.info(f"Performance Summary - {self.current_operation}:")
        self.logger.info(f"  Duration: {duration:.2f}s")
        self.logger.info(f"  Throughput: {throughput:.2f} MB/s")
        self.logger.info(f"  Memory Peak: {self.peak_memory:.1f} MB")
        self.logger.info(f"  CPU Average: {avg_cpu:.1f}%")
        self.logger.info(f"  Shards Created: {shards_created}")
        self.logger.info(f"  Compression Ratio: {compression_ratio:.2f}")
        
        # Reset state
        self.current_operation = None
        self.start_time = None
        self.start_memory = None
        
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of all performance metrics.
        
        Returns:
            Dictionary with performance summary
        """
        if not self.metrics_history:
            return {"total_operations": 0}
        
        total_duration = sum(m.duration for m in self.metrics_history)
        total_input = sum(m.input_size for m in self.metrics_history)
        total_output = sum(m.output_size for m in self.metrics_history)
        total_shards = sum(m.shards_created for m in self.metrics_history)
        
        avg_throughput = sum(m.throughput_mbps for m in self.metrics_history) / len(self.metrics_history)
        avg_memory = sum(m.memory_peak_mb for m in self.metrics_history) / len(self.metrics_history)
        avg_cpu = sum(m.cpu_percent for m in self.metrics_history) / len(self.metrics_history)
        
        return {
            "total_operations": len(self.metrics_history),
            "total_duration": total_duration,
            "total_input_mb": total_input / 1024 / 1024,
            "total_output_mb": total_output / 1024 / 1024,
            "total_shards_created": total_shards,
            "average_throughput_mbps": avg_throughput,
            "average_memory_peak_mb": avg_memory,
            "average_cpu_percent": avg_cpu,
            "overall_compression_ratio": total_output / total_input if total_input > 0 else 1.0,
            "operations": [
                {
                    "name": m.operation_name,
                    "duration": m.duration,
                    "throughput": m.throughput_mbps,
                    "memory_peak": m.memory_peak_mb,
                    "shards": m.shards_created
                }
                for m in self.metrics_history
            ]
        }
    
    def optimize_based_on_history(self) -> Dict[str, Any]:
        """
        Provide optimization recommendations based on performance history.
        
        Returns:
            Dictionary with optimization recommendations
        """
        if not self.metrics_history:
            return {"recommendations": ["No performance data available"]}
        
        recommendations = []
        
        # Analyze memory usage patterns
        avg_memory = sum(m.memory_peak_mb for m in self.metrics_history) / len(self.metrics_history)
        max_memory = max(m.memory_peak_mb for m in self.metrics_history)
        
        if max_memory > 2000:  # > 2GB
            recommendations.append("Consider reducing chunk size or enabling streaming for large datasets")
        
        if avg_memory > 1000:  # > 1GB average
            recommendations.append("High memory usage detected. Consider processing in smaller batches")
        
        # Analyze throughput patterns
        avg_throughput = sum(m.throughput_mbps for m in self.metrics_history) / len(self.metrics_history)
        
        if avg_throughput < 1.0:  # < 1 MB/s
            recommendations.append("Low throughput detected. Consider enabling parallel processing")
        
        # Analyze compression efficiency
        avg_compression = sum(m.compression_ratio for m in self.metrics_history) / len(self.metrics_history)
        
        if avg_compression > 1.5:  # Output larger than input
            recommendations.append("Poor compression ratio. Consider optimizing data structure or enabling compression")
        
        # Analyze shard distribution
        shard_counts = [m.shards_created for m in self.metrics_history]
        if shard_counts:
            avg_shards = sum(shard_counts) / len(shard_counts)
            if avg_shards > 100:
                recommendations.append("Many shards created. Consider increasing chunk size")
            elif avg_shards < 2:
                recommendations.append("Few shards created. Consider decreasing chunk size for better parallelization")
        
        return {
            "performance_summary": {
                "average_memory_mb": avg_memory,
                "max_memory_mb": max_memory,
                "average_throughput_mbps": avg_throughput,
                "average_compression_ratio": avg_compression,
                "average_shards": sum(shard_counts) / len(shard_counts) if shard_counts else 0
            },
            "recommendations": recommendations if recommendations else ["Performance looks optimal"]
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """
        Export performance metrics in specified format.
        
        Args:
            format: Export format ("json", "csv", "summary")
            
        Returns:
            Formatted metrics string
        """
        if format == "json":
            import json
            return json.dumps([
                {
                    "operation": m.operation_name,
                    "duration": m.duration,
                    "input_size": m.input_size,
                    "output_size": m.output_size,
                    "memory_peak_mb": m.memory_peak_mb,
                    "throughput_mbps": m.throughput_mbps,
                    "shards_created": m.shards_created,
                    "compression_ratio": m.compression_ratio
                }
                for m in self.metrics_history
            ], indent=2)
        
        elif format == "csv":
            lines = ["operation,duration,input_size,output_size,memory_peak_mb,throughput_mbps,shards_created,compression_ratio"]
            for m in self.metrics_history:
                lines.append(f"{m.operation_name},{m.duration},{m.input_size},{m.output_size},"
                           f"{m.memory_peak_mb},{m.throughput_mbps},{m.shards_created},{m.compression_ratio}")
            return "\n".join(lines)
        
        elif format == "summary":
            summary = self.get_performance_summary()
            lines = [
                f"Performance Summary:",
                f"  Total Operations: {summary['total_operations']}",
                f"  Total Duration: {summary['total_duration']:.2f}s",
                f"  Total Input: {summary['total_input_mb']:.2f} MB",
                f"  Total Output: {summary['total_output_mb']:.2f} MB",
                f"  Average Throughput: {summary['average_throughput_mbps']:.2f} MB/s",
                f"  Average Memory Peak: {summary['average_memory_peak_mb']:.1f} MB",
                f"  Total Shards Created: {summary['total_shards_created']}"
            ]
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global profiler instance for easy access
_global_profiler = PerformanceProfiler()

def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _global_profiler