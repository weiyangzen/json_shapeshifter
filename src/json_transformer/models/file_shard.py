"""File shard model implementation."""

from dataclasses import dataclass
from typing import Any, Dict
from .shard_metadata import ShardMetadata


@dataclass
class FileShard:
    """
    Represents a file shard containing data and associated metadata.
    
    A shard is a logical unit of the original JSON that has been split
    for size constraints while maintaining linking information.
    """
    
    id: str
    data: Any
    metadata: ShardMetadata
    size: int
    
    def __post_init__(self):
        """Validate shard after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate file shard integrity."""
        if not self.id:
            raise ValueError("id cannot be empty")
        
        if self.id != self.metadata.shard_id:
            raise ValueError("id must match metadata.shard_id")
        
        if self.size < 0:
            raise ValueError("size must be non-negative")
        
        if not self.id.startswith('shard_'):
            raise ValueError("id must start with 'shard_'")
    
    def to_file_schema(self) -> Dict[str, Any]:
        """
        Convert shard to file schema format for writing to disk.
        
        Returns:
            Dictionary in the standard file schema format
        """
        return {
            "_metadata": self.metadata.to_dict(),
            "data": self.data
        }
    
    @classmethod
    def from_file_schema(cls, schema: Dict[str, Any]) -> 'FileShard':
        """
        Create FileShard from file schema format.
        
        Args:
            schema: Dictionary in file schema format
            
        Returns:
            FileShard instance
        """
        metadata = ShardMetadata.from_dict(schema["_metadata"])
        
        # Calculate size (this will be recalculated more accurately later)
        import json
        estimated_size = len(json.dumps(schema, ensure_ascii=False).encode('utf-8'))
        
        return cls(
            id=metadata.shard_id,
            data=schema["data"],
            metadata=metadata,
            size=estimated_size
        )
    
    def get_size_kb(self) -> float:
        """Get shard size in kilobytes."""
        return self.size / 1024
    
    def is_oversized(self, max_size: int) -> bool:
        """Check if shard exceeds maximum size."""
        return self.size > max_size
    
    def has_nested_data(self) -> bool:
        """Check if shard contains nested dictionary or list data."""
        if isinstance(self.data, dict):
            return any(isinstance(v, (dict, list)) for v in self.data.values())
        elif isinstance(self.data, list):
            return any(isinstance(item, (dict, list)) for item in self.data)
        return False
    
    def get_data_summary(self) -> str:
        """Get a summary description of the data content."""
        if isinstance(self.data, dict):
            return f"dict with {len(self.data)} keys"
        elif isinstance(self.data, list):
            return f"list with {len(self.data)} items"
        else:
            return f"{type(self.data).__name__}: {str(self.data)[:50]}..."
    
    def clone(self) -> 'FileShard':
        """Create a deep copy of this shard."""
        import copy
        return FileShard(
            id=self.id,
            data=copy.deepcopy(self.data),
            metadata=self.metadata.clone(),
            size=self.size
        )