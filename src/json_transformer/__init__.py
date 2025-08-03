"""
JSON Transformer - Bidirectional JSON transformation tool.

Converts between compact von Neumann machine-readable JSON format 
and distributed LLM-readable file structures.
"""

from .json_transformer import JSONTransformer
from .models import FileShard, ShardMetadata, GlobalMetadata
from .types import UnflattenResult, FlattenResult

__version__ = "1.0.0"
__all__ = [
    "JSONTransformer",
    "FileShard", 
    "ShardMetadata",
    "GlobalMetadata",
    "UnflattenResult",
    "FlattenResult",
]