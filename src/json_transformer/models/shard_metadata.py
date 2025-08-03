"""Shard metadata model with validation and utility functions."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from ..types import DataType


@dataclass
class ShardMetadata:
    """
    Metadata for a file shard containing linking and structural information.
    
    This class represents the metadata that accompanies each shard file,
    providing the necessary information to reconstruct the original structure.
    """
    
    shard_id: str
    parent_id: Optional[str]
    child_ids: List[str]
    data_type: DataType
    original_path: List[str]
    next_shard: Optional[str] = None
    previous_shard: Optional[str] = None
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Validate metadata after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate metadata integrity."""
        if not self.shard_id:
            raise ValueError("shard_id cannot be empty")
        
        if not isinstance(self.child_ids, list):
            raise ValueError("child_ids must be a list")
        
        if not isinstance(self.original_path, list):
            raise ValueError("original_path must be a list")
        
        if self.data_type not in DataType:
            raise ValueError(f"Invalid data_type: {self.data_type}")
        
        # Validate shard ID format
        if not self.shard_id.startswith('shard_'):
            raise ValueError("shard_id must start with 'shard_'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            "shardId": self.shard_id,
            "parentId": self.parent_id,
            "childIds": self.child_ids,
            "dataType": self.data_type.value,
            "originalPath": self.original_path,
            "nextShard": self.next_shard,
            "previousShard": self.previous_shard,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShardMetadata':
        """Create ShardMetadata from dictionary."""
        return cls(
            shard_id=data["shardId"],
            parent_id=data.get("parentId"),
            child_ids=data.get("childIds", []),
            data_type=DataType(data["dataType"]),
            original_path=data.get("originalPath", []),
            next_shard=data.get("nextShard"),
            previous_shard=data.get("previousShard"),
            version=data.get("version", "1.0.0")
        )
    
    def add_child(self, child_id: str) -> None:
        """Add a child shard ID."""
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)
    
    def remove_child(self, child_id: str) -> None:
        """Remove a child shard ID."""
        if child_id in self.child_ids:
            self.child_ids.remove(child_id)
    
    def has_children(self) -> bool:
        """Check if this shard has child shards."""
        return len(self.child_ids) > 0
    
    def is_root(self) -> bool:
        """Check if this is a root shard."""
        return self.parent_id is None
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf shard (no children)."""
        return len(self.child_ids) == 0
    
    def get_path_string(self) -> str:
        """Get original path as dot-separated string."""
        return ".".join(self.original_path) if self.original_path else "root"
    
    def clone(self) -> 'ShardMetadata':
        """Create a deep copy of this metadata."""
        return ShardMetadata(
            shard_id=self.shard_id,
            parent_id=self.parent_id,
            child_ids=self.child_ids.copy(),
            data_type=self.data_type,
            original_path=self.original_path.copy(),
            next_shard=self.next_shard,
            previous_shard=self.previous_shard,
            version=self.version
        )