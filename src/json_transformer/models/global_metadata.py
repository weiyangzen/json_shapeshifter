"""Global metadata model for the entire transformation."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any
from ..types import DataStructure


@dataclass
class GlobalMetadata:
    """
    Global metadata for the entire JSON transformation operation.
    
    Contains high-level information about the transformation including
    statistics, structure type, and reconstruction information.
    """
    
    version: str
    original_size: int
    shard_count: int
    root_shard: str
    created_at: datetime
    data_structure: DataStructure
    
    def __post_init__(self):
        """Validate metadata after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate global metadata integrity."""
        if not self.version:
            raise ValueError("version cannot be empty")
        
        if self.original_size < 0:
            raise ValueError("original_size must be non-negative")
        
        if self.shard_count <= 0:
            raise ValueError("shard_count must be positive")
        
        if not self.root_shard:
            raise ValueError("root_shard cannot be empty")
        
        if not self.root_shard.startswith('shard_'):
            raise ValueError("root_shard must start with 'shard_'")
        
        if self.data_structure not in DataStructure:
            raise ValueError(f"Invalid data_structure: {self.data_structure}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "originalSize": self.original_size,
            "shardCount": self.shard_count,
            "rootShard": self.root_shard,
            "createdAt": self.created_at.isoformat(),
            "dataStructure": self.data_structure.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GlobalMetadata':
        """Create GlobalMetadata from dictionary."""
        return cls(
            version=data["version"],
            original_size=data["originalSize"],
            shard_count=data["shardCount"],
            root_shard=data["rootShard"],
            created_at=datetime.fromisoformat(data["createdAt"]),
            data_structure=DataStructure(data["dataStructure"])
        )
    
    def get_size_mb(self) -> float:
        """Get original size in megabytes."""
        return self.original_size / (1024 * 1024)
    
    def get_average_shard_size(self) -> float:
        """Get average shard size in bytes."""
        return self.original_size / self.shard_count if self.shard_count > 0 else 0
    
    def is_large_dataset(self, threshold_mb: float = 10.0) -> bool:
        """Check if this represents a large dataset."""
        return self.get_size_mb() > threshold_mb
    
    def get_creation_age_hours(self) -> float:
        """Get age of creation in hours."""
        now = datetime.now()
        if self.created_at.tzinfo is None and now.tzinfo is not None:
            # Make created_at timezone-aware if now is timezone-aware
            created_at = self.created_at.replace(tzinfo=now.tzinfo)
        else:
            created_at = self.created_at
        
        delta = now - created_at
        return delta.total_seconds() / 3600