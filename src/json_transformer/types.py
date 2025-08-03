"""Core type definitions for the JSON Transformer."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


class DataType(Enum):
    """Enumeration of supported data types."""
    DICT = "dict"
    LIST = "list"
    PRIMITIVE = "primitive"


class DataStructure(Enum):
    """Enumeration of data structure types."""
    DICT_OF_DICTS = "dict-of-dicts"
    LIST = "list"
    MIXED = "mixed"


class ErrorType(Enum):
    """Enumeration of error types."""
    SYNTAX = "syntax"
    STRUCTURE = "structure"
    SIZE = "size"
    PATH = "path"
    MEMORY = "memory"
    CIRCULAR = "circular"
    CORRUPTION = "corruption"
    FILESYSTEM = "filesystem"


@dataclass
class UnflattenResult:
    """Result of unflatten operation."""
    success: bool
    output_directory: str
    file_count: int
    total_size: int
    errors: Optional[List[str]] = None


@dataclass
class FlattenResult:
    """Result of flatten operation."""
    success: bool
    json_string: str
    errors: Optional[List[str]] = None


@dataclass
class ValidationError:
    """Validation error details."""
    type: ErrorType
    message: str
    location: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[str]


@dataclass
class ErrorResponse:
    """Response for error handling."""
    can_recover: bool
    suggested_action: str
    partial_results: Optional[Any] = None


@dataclass
class RecoveryResult:
    """Result of error recovery attempt."""
    recovered: bool
    recovered_shards: List['FileShard']
    lost_data: List[str]


@dataclass
class ShardMetadata:
    """Metadata for a file shard."""
    shard_id: str
    parent_id: Optional[str]
    child_ids: List[str]
    data_type: DataType
    original_path: List[str]
    next_shard: Optional[str] = None
    previous_shard: Optional[str] = None
    version: str = "1.0.0"


@dataclass
class GlobalMetadata:
    """Global metadata for the entire transformation."""
    version: str
    original_size: int
    shard_count: int
    root_shard: str
    created_at: datetime
    data_structure: DataStructure


@dataclass
class FileShard:
    """Represents a file shard with data and metadata."""
    id: str
    data: Any
    metadata: ShardMetadata
    size: int


@dataclass
class ProcessingResult:
    """Result of data processing."""
    shards: List[FileShard]
    root_shard_id: str
    metadata: GlobalMetadata


@dataclass
class FileSchema:
    """Schema structure for output files."""
    _metadata: Dict[str, Any]
    data: Any


class ProcessingError(Exception):
    """Custom exception for processing errors."""
    
    def __init__(self, message: str, error_type: ErrorType, context: Optional[Any] = None):
        super().__init__(message)
        self.error_type = error_type
        self.context = context


# Abstract base classes for interfaces

class JSONTransformerInterface(ABC):
    """Abstract interface for JSON Transformer."""
    
    @abstractmethod
    async def unflatten(
        self, 
        json_string: str, 
        max_size: Optional[int] = None, 
        output_dir: Optional[str] = None
    ) -> UnflattenResult:
        """Unflatten JSON string into multiple LLM-readable files."""
        pass
    
    @abstractmethod
    async def flatten(self, input_dir: str) -> FlattenResult:
        """Flatten directory of files back into single JSON string."""
        pass


class FileShardingEngineInterface(ABC):
    """Abstract interface for file sharding engine."""
    
    @abstractmethod
    def create_shard(self, data: Any, shard_id: str, metadata: ShardMetadata) -> FileShard:
        """Create a new file shard."""
        pass
    
    @abstractmethod
    def calculate_size(self, data: Any) -> int:
        """Calculate size of data in bytes."""
        pass
    
    @abstractmethod
    def should_create_new_shard(
        self, 
        current_size: int, 
        new_data_size: int, 
        max_size: int
    ) -> bool:
        """Determine if a new shard should be created."""
        pass


class DataProcessorInterface(ABC):
    """Abstract interface for data processors."""
    
    @abstractmethod
    def process(self, data: Any, max_size: int) -> ProcessingResult:
        """Process data into shards."""
        pass


class ErrorHandlerInterface(ABC):
    """Abstract interface for error handling."""
    
    @abstractmethod
    def validate_input(self, input_data: str) -> ValidationResult:
        """Validate input data."""
        pass
    
    @abstractmethod
    def handle_processing_error(self, error: ProcessingError) -> ErrorResponse:
        """Handle processing errors."""
        pass
    
    @abstractmethod
    def recover_from_corruption(self, shards: List[FileShard]) -> RecoveryResult:
        """Attempt to recover from corrupted data."""
        pass