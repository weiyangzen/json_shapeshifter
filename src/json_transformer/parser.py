"""JSON parser with validation and data type detection."""

import json
import logging
from typing import Any, Dict, List, Union, Optional, Tuple
from .types import DataStructure, ValidationResult, ValidationError, ErrorType
from .error_handler import ErrorHandler


class JSONParser:
    """
    JSON parser with comprehensive validation and data type detection.
    
    Provides parsing capabilities with structure analysis to determine
    the best processing strategy for different JSON data types.
    """
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None, 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the JSON parser.
        
        Args:
            error_handler: Optional ErrorHandler instance
            logger: Optional logger instance
        """
        self.error_handler = error_handler or ErrorHandler()
        self.logger = logger or logging.getLogger(__name__)
    
    def parse(self, json_string: str) -> Tuple[Any, DataStructure]:
        """
        Parse JSON string and detect data structure type.
        
        Args:
            json_string: JSON string to parse
            
        Returns:
            Tuple of (parsed_data, data_structure_type)
            
        Raises:
            ValueError: If JSON is invalid or unsupported
        """
        # Validate input
        validation_result = self.error_handler.validate_input(json_string)
        if not validation_result.is_valid:
            error_messages = [error.message for error in validation_result.errors]
            raise ValueError(f"Invalid JSON input: {'; '.join(error_messages)}")
        
        # Parse JSON
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing failed: {e.msg} at line {e.lineno}, column {e.colno}")
        
        # Detect data structure type
        structure_type = self.detect_data_structure(data)
        
        self.logger.info(f"Parsed JSON with structure type: {structure_type.value}")
        return data, structure_type
    
    def detect_data_structure(self, data: Any) -> DataStructure:
        """
        Detect the data structure type of parsed JSON.
        
        Args:
            data: Parsed JSON data
            
        Returns:
            DataStructure enum indicating the structure type
        """
        if isinstance(data, dict):
            return self._analyze_dict_structure(data)
        elif isinstance(data, list):
            return self._analyze_list_structure(data)
        else:
            # Primitive types are not supported as root elements
            raise ValueError(f"Unsupported root data type: {type(data).__name__}")
    
    def _analyze_dict_structure(self, data: Dict[str, Any]) -> DataStructure:
        """
        Analyze dictionary structure to determine type.
        
        Args:
            data: Dictionary data to analyze
            
        Returns:
            DataStructure indicating dict type
        """
        if not data:
            return DataStructure.DICT_OF_DICTS  # Empty dict defaults to dict-of-dicts
        
        # Count different value types
        dict_values = 0
        list_values = 0
        primitive_values = 0
        
        for value in data.values():
            if isinstance(value, dict):
                dict_values += 1
            elif isinstance(value, list):
                list_values += 1
            else:
                primitive_values += 1
        
        # Determine structure based on value types
        total_values = len(data)
        dict_ratio = dict_values / total_values
        list_ratio = list_values / total_values
        
        if dict_ratio >= 0.7:  # 70% or more are dictionaries
            return DataStructure.DICT_OF_DICTS
        elif dict_ratio > 0 or list_ratio > 0:  # Mixed content
            return DataStructure.MIXED
        else:  # All primitive values
            return DataStructure.DICT_OF_DICTS  # Treat as simple dict-of-dicts
    
    def _analyze_list_structure(self, data: List[Any]) -> DataStructure:
        """
        Analyze list structure to determine type.
        
        Args:
            data: List data to analyze
            
        Returns:
            DataStructure indicating list type
        """
        if not data:
            return DataStructure.LIST  # Empty list
        
        # Check if list contains mixed types
        has_dicts = any(isinstance(item, dict) for item in data)
        has_lists = any(isinstance(item, list) for item in data)
        has_primitives = any(not isinstance(item, (dict, list)) for item in data)
        
        if has_dicts and (has_lists or has_primitives):
            return DataStructure.MIXED
        elif has_lists and has_primitives:
            return DataStructure.MIXED
        else:
            return DataStructure.LIST
    
    def validate_for_processing(self, data: Any, max_size: int) -> ValidationResult:
        """
        Validate data for processing with size constraints.
        
        Args:
            data: Parsed data to validate
            max_size: Maximum allowed size for processing
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        
        # Calculate data size
        try:
            json_string = json.dumps(data, ensure_ascii=False)
            data_size = len(json_string.encode('utf-8'))
        except (TypeError, ValueError) as e:
            errors.append(ValidationError(
                type=ErrorType.STRUCTURE,
                message=f"Data is not JSON serializable: {str(e)}",
                location="data"
            ))
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Check size constraints
        if data_size > max_size * 1000:  # If data is 1000x larger than max_size
            warnings.append(f"Data size ({data_size} bytes) is much larger than max shard size "
                          f"({max_size} bytes). This will create many shards.")
        
        # Check nesting depth
        max_depth = self._calculate_nesting_depth(data)
        if max_depth > 10:
            warnings.append(f"Deep nesting detected (depth: {max_depth}). "
                          "This may impact processing performance.")
        
        # Check for potential circular references
        if self._has_potential_circular_refs(data):
            errors.append(ValidationError(
                type=ErrorType.CIRCULAR,
                message="Potential circular references detected in data structure",
                location="data"
            ))
        
        # Check for very large individual values
        large_values = self._find_large_values(data, max_size // 2)
        if large_values:
            warnings.append(f"Found {len(large_values)} large values that may not fit in single shards")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _calculate_nesting_depth(self, data: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth of data structure."""
        if not isinstance(data, (dict, list)):
            return current_depth
        
        max_child_depth = current_depth
        
        if isinstance(data, dict):
            for value in data.values():
                child_depth = self._calculate_nesting_depth(value, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
        else:  # list
            for item in data:
                child_depth = self._calculate_nesting_depth(item, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def _has_potential_circular_refs(self, data: Any, seen_ids: Optional[set] = None) -> bool:
        """Check for potential circular references using object IDs."""
        if seen_ids is None:
            seen_ids = set()
        
        if isinstance(data, (dict, list)):
            obj_id = id(data)
            if obj_id in seen_ids:
                return True
            
            seen_ids.add(obj_id)
            
            try:
                if isinstance(data, dict):
                    for value in data.values():
                        if self._has_potential_circular_refs(value, seen_ids):
                            return True
                else:  # list
                    for item in data:
                        if self._has_potential_circular_refs(item, seen_ids):
                            return True
            finally:
                seen_ids.remove(obj_id)
        
        return False
    
    def _find_large_values(self, data: Any, size_threshold: int, 
                          path: List[str] = None) -> List[Tuple[List[str], int]]:
        """Find values that exceed size threshold."""
        if path is None:
            path = []
        
        large_values = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = path + [str(key)]
                
                # Check current value size
                try:
                    value_size = len(json.dumps(value, ensure_ascii=False).encode('utf-8'))
                    if value_size > size_threshold:
                        large_values.append((current_path, value_size))
                except (TypeError, ValueError):
                    pass  # Skip non-serializable values
                
                # Recurse into nested structures
                if isinstance(value, (dict, list)):
                    large_values.extend(self._find_large_values(value, size_threshold, current_path))
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = path + [f"[{i}]"]
                
                # Check current item size
                try:
                    item_size = len(json.dumps(item, ensure_ascii=False).encode('utf-8'))
                    if item_size > size_threshold:
                        large_values.append((current_path, item_size))
                except (TypeError, ValueError):
                    pass  # Skip non-serializable items
                
                # Recurse into nested structures
                if isinstance(item, (dict, list)):
                    large_values.extend(self._find_large_values(item, size_threshold, current_path))
        
        return large_values
    
    def get_structure_statistics(self, data: Any) -> Dict[str, Any]:
        """
        Get detailed statistics about the data structure.
        
        Args:
            data: Parsed data to analyze
            
        Returns:
            Dictionary with structure statistics
        """
        stats = {
            "total_size": 0,
            "max_depth": 0,
            "dict_count": 0,
            "list_count": 0,
            "primitive_count": 0,
            "total_keys": 0,
            "total_items": 0,
            "structure_type": self.detect_data_structure(data).value
        }
        
        try:
            stats["total_size"] = len(json.dumps(data, ensure_ascii=False).encode('utf-8'))
        except (TypeError, ValueError):
            stats["total_size"] = -1  # Indicate error
        
        stats["max_depth"] = self._calculate_nesting_depth(data)
        
        # Count different element types
        self._count_elements(data, stats)
        
        return stats
    
    def _count_elements(self, data: Any, stats: Dict[str, Any]) -> None:
        """Recursively count different types of elements."""
        if isinstance(data, dict):
            stats["dict_count"] += 1
            stats["total_keys"] += len(data)
            
            for value in data.values():
                self._count_elements(value, stats)
        
        elif isinstance(data, list):
            stats["list_count"] += 1
            stats["total_items"] += len(data)
            
            for item in data:
                self._count_elements(item, stats)
        
        else:
            stats["primitive_count"] += 1