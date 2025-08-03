"""Advanced data type detection for JSON structures."""

import logging
from typing import Any, Dict, List, Tuple, Optional, Set
from collections import Counter
from .types import DataStructure, DataType


class DataTypeDetector:
    """
    Advanced data type detector for JSON structures.
    
    Provides sophisticated analysis of JSON data to determine the most
    appropriate processing strategy based on structure patterns.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the data type detector.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def detect_structure_type(self, data: Any) -> DataStructure:
        """
        Detect the overall structure type of JSON data.
        
        Args:
            data: Parsed JSON data
            
        Returns:
            DataStructure enum indicating the detected type
        """
        if isinstance(data, dict):
            return self._analyze_dict_structure(data)
        elif isinstance(data, list):
            return self._analyze_list_structure(data)
        else:
            raise ValueError(f"Unsupported root data type: {type(data).__name__}")
    
    def detect_element_type(self, element: Any) -> DataType:
        """
        Detect the type of a single element.
        
        Args:
            element: Element to analyze
            
        Returns:
            DataType enum indicating the element type
        """
        if isinstance(element, dict):
            return DataType.DICT
        elif isinstance(element, list):
            return DataType.LIST
        else:
            return DataType.PRIMITIVE
    
    def analyze_dict_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze patterns in dictionary structure.
        
        Args:
            data: Dictionary to analyze
            
        Returns:
            Dictionary with pattern analysis results
        """
        if not data:
            return {
                "is_empty": True,
                "structure_type": DataStructure.DICT_OF_DICTS,
                "value_types": {},
                "nesting_levels": 0,
                "homogeneous": True
            }
        
        # Analyze value types
        value_types = Counter()
        nesting_levels = []
        
        for key, value in data.items():
            element_type = self.detect_element_type(value)
            value_types[element_type.value] += 1
            
            # Calculate nesting level for this value
            nesting_level = self._calculate_element_depth(value)
            nesting_levels.append(nesting_level)
        
        # Determine homogeneity
        is_homogeneous = len(value_types) == 1
        
        # Determine structure type based on patterns
        total_values = len(data)
        dict_ratio = value_types.get(DataType.DICT.value, 0) / total_values
        list_ratio = value_types.get(DataType.LIST.value, 0) / total_values
        primitive_ratio = value_types.get(DataType.PRIMITIVE.value, 0) / total_values
        
        if dict_ratio >= 0.8:  # 80% or more are dictionaries
            structure_type = DataStructure.DICT_OF_DICTS
        elif primitive_ratio >= 0.8:  # 80% or more are primitives
            structure_type = DataStructure.DICT_OF_DICTS  # Still treat as dict-of-dicts
        else:
            structure_type = DataStructure.MIXED
        
        return {
            "is_empty": False,
            "structure_type": structure_type,
            "value_types": dict(value_types),
            "nesting_levels": {
                "min": min(nesting_levels) if nesting_levels else 0,
                "max": max(nesting_levels) if nesting_levels else 0,
                "avg": sum(nesting_levels) / len(nesting_levels) if nesting_levels else 0
            },
            "homogeneous": is_homogeneous,
            "dict_ratio": dict_ratio,
            "list_ratio": list_ratio,
            "primitive_ratio": primitive_ratio,
            "total_keys": len(data)
        }
    
    def analyze_list_patterns(self, data: List[Any]) -> Dict[str, Any]:
        """
        Analyze patterns in list structure.
        
        Args:
            data: List to analyze
            
        Returns:
            Dictionary with pattern analysis results
        """
        if not data:
            return {
                "is_empty": True,
                "structure_type": DataStructure.LIST,
                "item_types": {},
                "nesting_levels": 0,
                "homogeneous": True
            }
        
        # Analyze item types
        item_types = Counter()
        nesting_levels = []
        
        for item in data:
            element_type = self.detect_element_type(item)
            item_types[element_type.value] += 1
            
            # Calculate nesting level for this item
            nesting_level = self._calculate_element_depth(item)
            nesting_levels.append(nesting_level)
        
        # Determine homogeneity
        is_homogeneous = len(item_types) == 1
        
        # Determine structure type based on patterns
        total_items = len(data)
        dict_ratio = item_types.get(DataType.DICT.value, 0) / total_items
        list_ratio = item_types.get(DataType.LIST.value, 0) / total_items
        primitive_ratio = item_types.get(DataType.PRIMITIVE.value, 0) / total_items
        
        if is_homogeneous:
            structure_type = DataStructure.LIST
        else:
            structure_type = DataStructure.MIXED
        
        return {
            "is_empty": False,
            "structure_type": structure_type,
            "item_types": dict(item_types),
            "nesting_levels": {
                "min": min(nesting_levels) if nesting_levels else 0,
                "max": max(nesting_levels) if nesting_levels else 0,
                "avg": sum(nesting_levels) / len(nesting_levels) if nesting_levels else 0
            },
            "homogeneous": is_homogeneous,
            "dict_ratio": dict_ratio,
            "list_ratio": list_ratio,
            "primitive_ratio": primitive_ratio,
            "total_items": len(data)
        }
    
    def detect_nested_structures(self, data: Any, path: List[str] = None) -> List[Dict[str, Any]]:
        """
        Detect all nested structures in the data.
        
        Args:
            data: Data to analyze
            path: Current path in the data structure
            
        Returns:
            List of nested structure information
        """
        if path is None:
            path = []
        
        nested_structures = []
        
        if isinstance(data, dict):
            # Add current dict as a nested structure
            nested_structures.append({
                "path": path.copy(),
                "type": DataType.DICT,
                "size": len(data),
                "depth": len(path)
            })
            
            # Recurse into values
            for key, value in data.items():
                current_path = path + [str(key)]
                nested_structures.extend(
                    self.detect_nested_structures(value, current_path)
                )
        
        elif isinstance(data, list):
            # Add current list as a nested structure
            nested_structures.append({
                "path": path.copy(),
                "type": DataType.LIST,
                "size": len(data),
                "depth": len(path)
            })
            
            # Recurse into items
            for i, item in enumerate(data):
                current_path = path + [f"[{i}]"]
                nested_structures.extend(
                    self.detect_nested_structures(item, current_path)
                )
        
        return nested_structures
    
    def identify_sharding_candidates(self, data: Any, size_threshold: int) -> List[Dict[str, Any]]:
        """
        Identify structures that are candidates for sharding.
        
        Args:
            data: Data to analyze
            size_threshold: Size threshold for sharding consideration
            
        Returns:
            List of sharding candidate information
        """
        import json
        
        candidates = []
        nested_structures = self.detect_nested_structures(data)
        
        for structure in nested_structures:
            # Get the actual data at this path
            structure_data = self._get_data_at_path(data, structure["path"])
            
            if structure_data is not None:
                try:
                    # Calculate serialized size
                    serialized_size = len(
                        json.dumps(structure_data, ensure_ascii=False).encode('utf-8')
                    )
                    
                    # Check if it's a sharding candidate
                    if serialized_size > size_threshold:
                        candidates.append({
                            "path": structure["path"],
                            "type": structure["type"],
                            "size": serialized_size,
                            "depth": structure["depth"],
                            "element_count": structure["size"],
                            "sharding_priority": self._calculate_sharding_priority(
                                structure, serialized_size, size_threshold
                            )
                        })
                
                except (TypeError, ValueError):
                    # Skip non-serializable structures
                    continue
        
        # Sort by sharding priority (higher priority first)
        candidates.sort(key=lambda x: x["sharding_priority"], reverse=True)
        
        return candidates
    
    def _analyze_dict_structure(self, data: Dict[str, Any]) -> DataStructure:
        """Analyze dictionary structure to determine type."""
        patterns = self.analyze_dict_patterns(data)
        return patterns["structure_type"]
    
    def _analyze_list_structure(self, data: List[Any]) -> DataStructure:
        """Analyze list structure to determine type."""
        patterns = self.analyze_list_patterns(data)
        return patterns["structure_type"]
    
    def _calculate_element_depth(self, element: Any, current_depth: int = 0) -> int:
        """Calculate the maximum depth of an element."""
        if not isinstance(element, (dict, list)):
            return current_depth
        
        max_child_depth = current_depth
        
        if isinstance(element, dict):
            for value in element.values():
                child_depth = self._calculate_element_depth(value, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
        else:  # list
            for item in element:
                child_depth = self._calculate_element_depth(item, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def _get_data_at_path(self, data: Any, path: List[str]) -> Any:
        """Get data at a specific path."""
        current = data
        
        for segment in path:
            try:
                if isinstance(current, dict):
                    current = current[segment]
                elif isinstance(current, list):
                    # Handle list indices like "[0]"
                    if segment.startswith('[') and segment.endswith(']'):
                        index = int(segment[1:-1])
                        current = current[index]
                    else:
                        return None
                else:
                    return None
            except (KeyError, IndexError, ValueError, TypeError):
                return None
        
        return current
    
    def _calculate_sharding_priority(self, structure: Dict[str, Any], 
                                   size: int, threshold: int) -> float:
        """
        Calculate sharding priority for a structure.
        
        Higher priority means should be sharded first.
        """
        # Base priority on size ratio
        size_ratio = size / threshold
        
        # Adjust based on depth (deeper structures get lower priority)
        depth_penalty = structure["depth"] * 0.1
        
        # Adjust based on element count (more elements = higher priority)
        element_bonus = min(structure["size"] / 100, 1.0)  # Cap at 1.0
        
        # Adjust based on type (dicts get higher priority than lists)
        type_bonus = 0.2 if structure["type"] == DataType.DICT else 0.0
        
        priority = size_ratio + element_bonus + type_bonus - depth_penalty
        
        return max(priority, 0.1)  # Minimum priority of 0.1
    
    def get_processing_recommendations(self, data: Any, max_size: int) -> Dict[str, Any]:
        """
        Get recommendations for processing the data structure.
        
        Args:
            data: Data to analyze
            max_size: Maximum size constraint
            
        Returns:
            Dictionary with processing recommendations
        """
        structure_type = self.detect_structure_type(data)
        
        if isinstance(data, dict):
            patterns = self.analyze_dict_patterns(data)
        else:
            patterns = self.analyze_list_patterns(data)
        
        sharding_candidates = self.identify_sharding_candidates(data, max_size)
        
        recommendations = {
            "structure_type": structure_type.value,
            "recommended_processor": self._recommend_processor(structure_type),
            "sharding_needed": len(sharding_candidates) > 0,
            "estimated_shards": self._estimate_shard_count(data, max_size),
            "complexity_score": self._calculate_complexity_score(patterns),
            "processing_strategy": self._recommend_processing_strategy(
                structure_type, patterns, sharding_candidates
            ),
            "sharding_candidates": sharding_candidates[:5],  # Top 5 candidates
            "patterns": patterns
        }
        
        return recommendations
    
    def _recommend_processor(self, structure_type: DataStructure) -> str:
        """Recommend the appropriate processor for the structure type."""
        if structure_type == DataStructure.DICT_OF_DICTS:
            return "DictProcessor"
        elif structure_type == DataStructure.LIST:
            return "ListProcessor"
        else:
            return "MixedProcessor"
    
    def _estimate_shard_count(self, data: Any, max_size: int) -> int:
        """Estimate the number of shards needed."""
        import json
        
        try:
            total_size = len(json.dumps(data, ensure_ascii=False).encode('utf-8'))
            estimated_shards = max(1, (total_size + max_size - 1) // max_size)  # Ceiling division
            return estimated_shards
        except (TypeError, ValueError):
            return 1  # Default to 1 if size can't be calculated
    
    def _calculate_complexity_score(self, patterns: Dict[str, Any]) -> float:
        """Calculate a complexity score for the data structure."""
        score = 0.0
        
        # Base score on nesting levels
        if "nesting_levels" in patterns:
            max_nesting = patterns["nesting_levels"].get("max", 0)
            score += max_nesting * 0.3
        
        # Add score for heterogeneity
        if not patterns.get("homogeneous", True):
            score += 0.5
        
        # Add score for mixed types
        if patterns.get("structure_type") == DataStructure.MIXED:
            score += 0.7
        
        # Add score based on size
        total_elements = patterns.get("total_keys", 0) + patterns.get("total_items", 0)
        if total_elements > 100:
            score += 0.4
        elif total_elements > 1000:
            score += 0.8
        
        return min(score, 5.0)  # Cap at 5.0
    
    def _recommend_processing_strategy(self, structure_type: DataStructure, 
                                     patterns: Dict[str, Any], 
                                     sharding_candidates: List[Dict[str, Any]]) -> str:
        """Recommend a processing strategy based on analysis."""
        if not sharding_candidates:
            return "simple_processing"
        
        complexity_score = self._calculate_complexity_score(patterns)
        
        if complexity_score > 3.0:
            return "complex_hierarchical_sharding"
        elif len(sharding_candidates) > 10:
            return "aggressive_sharding"
        elif structure_type == DataStructure.MIXED:
            return "mixed_strategy_sharding"
        else:
            return "standard_sharding"