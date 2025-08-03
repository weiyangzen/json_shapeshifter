"""Validation utilities for metadata and data integrity."""

import json
from typing import List, Dict, Any, Set
from ..types import ValidationResult, ValidationError, ErrorType
from ..models import FileShard, ShardMetadata, GlobalMetadata


class ValidationUtils:
    """Utility class for validating data structures and metadata."""
    
    @staticmethod
    def validate_json_string(json_string: str) -> ValidationResult:
        """
        Validate JSON string syntax and structure.
        
        Args:
            json_string: JSON string to validate
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        
        # Check if string is empty
        if not json_string.strip():
            errors.append(ValidationError(
                type=ErrorType.SYNTAX,
                message="JSON string is empty",
                location="input"
            ))
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Try to parse JSON
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            errors.append(ValidationError(
                type=ErrorType.SYNTAX,
                message=f"Invalid JSON syntax: {e.msg}",
                location=f"line {e.lineno}, column {e.colno}"
            ))
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Validate structure
        structure_errors, structure_warnings = ValidationUtils._validate_json_structure(data)
        errors.extend(structure_errors)
        warnings.extend(structure_warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @staticmethod
    def _validate_json_structure(data: Any) -> tuple[List[ValidationError], List[str]]:
        """Validate JSON data structure."""
        errors = []
        warnings = []
        
        # Check for supported root types
        if not isinstance(data, (dict, list)):
            errors.append(ValidationError(
                type=ErrorType.STRUCTURE,
                message=f"Root element must be dict or list, got {type(data).__name__}",
                location="root"
            ))
            return errors, warnings
        
        # Check for circular references
        if ValidationUtils._has_circular_references(data):
            errors.append(ValidationError(
                type=ErrorType.STRUCTURE,
                message="Circular references detected in JSON structure",
                location="unknown"
            ))
        
        # Check depth
        max_depth = ValidationUtils._calculate_max_depth(data)
        if max_depth > 20:
            warnings.append(f"Deep nesting detected (depth: {max_depth}). This may impact performance.")
        
        return errors, warnings
    
    @staticmethod
    def _has_circular_references(data: Any, seen: Set[int] = None) -> bool:
        """Check for circular references in data structure."""
        if seen is None:
            seen = set()
        
        if isinstance(data, (dict, list)):
            obj_id = id(data)
            if obj_id in seen:
                return True
            seen.add(obj_id)
            
            if isinstance(data, dict):
                for value in data.values():
                    if ValidationUtils._has_circular_references(value, seen):
                        return True
            else:  # list
                for item in data:
                    if ValidationUtils._has_circular_references(item, seen):
                        return True
            
            seen.remove(obj_id)
        
        return False
    
    @staticmethod
    def _calculate_max_depth(data: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        if not isinstance(data, (dict, list)):
            return current_depth
        
        max_child_depth = current_depth
        
        if isinstance(data, dict):
            for value in data.values():
                child_depth = ValidationUtils._calculate_max_depth(value, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
        else:  # list
            for item in data:
                child_depth = ValidationUtils._calculate_max_depth(item, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    @staticmethod
    def validate_shard_metadata(metadata: ShardMetadata) -> ValidationResult:
        """
        Validate shard metadata integrity.
        
        Args:
            metadata: ShardMetadata to validate
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        
        try:
            # This will call the internal validation
            metadata._validate()
        except ValueError as e:
            errors.append(ValidationError(
                type=ErrorType.STRUCTURE,
                message=str(e),
                location="metadata"
            ))
        
        # Additional validations
        if metadata.parent_id and metadata.parent_id == metadata.shard_id:
            errors.append(ValidationError(
                type=ErrorType.STRUCTURE,
                message="Shard cannot be its own parent",
                location="parentId"
            ))
        
        if metadata.shard_id in metadata.child_ids:
            errors.append(ValidationError(
                type=ErrorType.STRUCTURE,
                message="Shard cannot be its own child",
                location="childIds"
            ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @staticmethod
    def validate_shard_links(shards: List[FileShard]) -> ValidationResult:
        """
        Validate linking integrity across multiple shards.
        
        Args:
            shards: List of FileShard objects to validate
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        
        if not shards:
            errors.append(ValidationError(
                type=ErrorType.STRUCTURE,
                message="No shards provided for validation",
                location="shards"
            ))
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Create lookup maps
        shard_map = {shard.id: shard for shard in shards}
        
        # Validate each shard's links
        for shard in shards:
            # Check parent links
            if shard.metadata.parent_id:
                if shard.metadata.parent_id not in shard_map:
                    errors.append(ValidationError(
                        type=ErrorType.STRUCTURE,
                        message=f"Parent shard '{shard.metadata.parent_id}' not found",
                        location=f"shard {shard.id}"
                    ))
            
            # Check child links
            for child_id in shard.metadata.child_ids:
                if child_id not in shard_map:
                    errors.append(ValidationError(
                        type=ErrorType.STRUCTURE,
                        message=f"Child shard '{child_id}' not found",
                        location=f"shard {shard.id}"
                    ))
                else:
                    # Verify bidirectional link
                    child_shard = shard_map[child_id]
                    if child_shard.metadata.parent_id != shard.id:
                        errors.append(ValidationError(
                            type=ErrorType.STRUCTURE,
                            message=f"Bidirectional link broken between {shard.id} and {child_id}",
                            location=f"shard {shard.id}"
                        ))
            
            # Check sequential links
            if shard.metadata.next_shard:
                if shard.metadata.next_shard not in shard_map:
                    errors.append(ValidationError(
                        type=ErrorType.STRUCTURE,
                        message=f"Next shard '{shard.metadata.next_shard}' not found",
                        location=f"shard {shard.id}"
                    ))
                else:
                    next_shard = shard_map[shard.metadata.next_shard]
                    if next_shard.metadata.previous_shard != shard.id:
                        errors.append(ValidationError(
                            type=ErrorType.STRUCTURE,
                            message=f"Sequential link broken between {shard.id} and {shard.metadata.next_shard}",
                            location=f"shard {shard.id}"
                        ))
        
        # Check for orphaned shards
        root_shards = [s for s in shards if s.metadata.is_root()]
        if len(root_shards) == 0:
            errors.append(ValidationError(
                type=ErrorType.STRUCTURE,
                message="No root shard found",
                location="shards"
            ))
        elif len(root_shards) > 1:
            warnings.append(f"Multiple root shards found: {[s.id for s in root_shards]}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )