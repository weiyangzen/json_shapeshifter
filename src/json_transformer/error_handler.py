"""Error handling implementation for the JSON Transformer."""

import logging
from typing import List, Optional, Any
from .types import (
    ErrorHandlerInterface,
    ValidationResult,
    ValidationError,
    ErrorResponse,
    RecoveryResult,
    ProcessingError,
    ErrorType
)
from .models import FileShard
from .utils.validation import ValidationUtils


class ErrorHandler(ErrorHandlerInterface):
    """
    Comprehensive error handler for JSON Transformer operations.
    
    Provides validation, error handling, and recovery capabilities
    for various error scenarios that can occur during processing.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the error handler.
        
        Args:
            logger: Optional logger instance for error reporting
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_input(self, input_data: str) -> ValidationResult:
        """
        Validate input JSON string.
        
        Args:
            input_data: JSON string to validate
            
        Returns:
            ValidationResult with validation details
        """
        try:
            return ValidationUtils.validate_json_string(input_data)
        except Exception as e:
            self.logger.error(f"Unexpected error during input validation: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[ValidationError(
                    type=ErrorType.SYNTAX,
                    message=f"Validation failed with unexpected error: {str(e)}",
                    location="input"
                )],
                warnings=[]
            )
    
    def handle_processing_error(self, error: ProcessingError) -> ErrorResponse:
        """
        Handle processing errors and provide recovery suggestions.
        
        Args:
            error: ProcessingError to handle
            
        Returns:
            ErrorResponse with recovery information
        """
        self.logger.error(f"Processing error: {error.error_type.value} - {error}")
        
        if error.error_type == ErrorType.MEMORY:
            return self._handle_memory_error(error)
        elif error.error_type == ErrorType.CIRCULAR:
            return self._handle_circular_error(error)
        elif error.error_type == ErrorType.CORRUPTION:
            return self._handle_corruption_error(error)
        elif error.error_type == ErrorType.FILESYSTEM:
            return self._handle_filesystem_error(error)
        else:
            return ErrorResponse(
                can_recover=False,
                suggested_action="Unknown error type. Please check logs and retry.",
                partial_results=None
            )
    
    def recover_from_corruption(self, shards: List[FileShard]) -> RecoveryResult:
        """
        Attempt to recover from corrupted shard data.
        
        Args:
            shards: List of potentially corrupted shards
            
        Returns:
            RecoveryResult with recovery status and recovered data
        """
        self.logger.info(f"Attempting to recover from corruption in {len(shards)} shards")
        
        recovered_shards = []
        lost_data = []
        
        for shard in shards:
            try:
                # Validate shard metadata
                validation_result = ValidationUtils.validate_shard_metadata(shard.metadata)
                
                if validation_result.is_valid:
                    # Shard metadata is valid, check data integrity
                    if self._validate_shard_data(shard):
                        recovered_shards.append(shard)
                    else:
                        lost_data.append(f"Shard {shard.id}: corrupted data")
                        # Try to create a minimal recovery shard
                        recovery_shard = self._create_recovery_shard(shard)
                        if recovery_shard:
                            recovered_shards.append(recovery_shard)
                else:
                    lost_data.append(f"Shard {shard.id}: corrupted metadata")
                    
            except Exception as e:
                self.logger.error(f"Error during recovery of shard {shard.id}: {e}")
                lost_data.append(f"Shard {shard.id}: recovery failed - {str(e)}")
        
        # Validate recovered shard links
        if recovered_shards:
            link_validation = ValidationUtils.validate_shard_links(recovered_shards)
            if not link_validation.is_valid:
                self.logger.warning("Recovered shards have broken links, attempting repair")
                recovered_shards = self._repair_shard_links(recovered_shards)
        
        recovery_success = len(recovered_shards) > 0
        
        self.logger.info(f"Recovery completed: {len(recovered_shards)} shards recovered, "
                        f"{len(lost_data)} items lost")
        
        return RecoveryResult(
            recovered=recovery_success,
            recovered_shards=recovered_shards,
            lost_data=lost_data
        )
    
    def _handle_memory_error(self, error: ProcessingError) -> ErrorResponse:
        """Handle memory-related errors."""
        return ErrorResponse(
            can_recover=True,
            suggested_action="Reduce file size limit or process in smaller chunks. "
                           "Consider increasing available memory or using streaming processing.",
            partial_results=error.context.get('partial_results') if error.context else None
        )
    
    def _handle_circular_error(self, error: ProcessingError) -> ErrorResponse:
        """Handle circular reference errors."""
        return ErrorResponse(
            can_recover=False,
            suggested_action="Remove circular references from input data. "
                           "Check for objects that reference themselves or create reference loops.",
            partial_results=None
        )
    
    def _handle_corruption_error(self, error: ProcessingError) -> ErrorResponse:
        """Handle data corruption errors."""
        return ErrorResponse(
            can_recover=True,
            suggested_action="Attempt data recovery using the recover_from_corruption method. "
                           "Some data may be lost but partial recovery may be possible.",
            partial_results=error.context.get('corrupted_shards') if error.context else None
        )
    
    def _handle_filesystem_error(self, error: ProcessingError) -> ErrorResponse:
        """Handle filesystem-related errors."""
        return ErrorResponse(
            can_recover=True,
            suggested_action="Check file permissions, available disk space, and directory access. "
                           "Ensure the output directory is writable and has sufficient space.",
            partial_results=error.context.get('partial_files') if error.context else None
        )
    
    def _validate_shard_data(self, shard: FileShard) -> bool:
        """
        Validate shard data integrity.
        
        Args:
            shard: FileShard to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check if data is serializable
            import json
            json.dumps(shard.data, ensure_ascii=False)
            
            # Check data type consistency
            if shard.metadata.data_type.value == "dict" and not isinstance(shard.data, dict):
                return False
            elif shard.metadata.data_type.value == "list" and not isinstance(shard.data, list):
                return False
            
            return True
            
        except (TypeError, ValueError, RecursionError):
            return False
    
    def _create_recovery_shard(self, corrupted_shard: FileShard) -> Optional[FileShard]:
        """
        Create a minimal recovery shard from corrupted data.
        
        Args:
            corrupted_shard: Corrupted FileShard
            
        Returns:
            Recovery FileShard or None if recovery not possible
        """
        try:
            # Create minimal data based on metadata type
            if corrupted_shard.metadata.data_type.value == "dict":
                recovery_data = {"_recovered": True, "_original_id": corrupted_shard.id}
            elif corrupted_shard.metadata.data_type.value == "list":
                recovery_data = [{"_recovered": True, "_original_id": corrupted_shard.id}]
            else:
                recovery_data = f"_recovered_from_{corrupted_shard.id}"
            
            # Create recovery shard with minimal data
            recovery_shard = FileShard(
                id=corrupted_shard.id,
                data=recovery_data,
                metadata=corrupted_shard.metadata,
                size=len(str(recovery_data))
            )
            
            return recovery_shard
            
        except Exception as e:
            self.logger.error(f"Failed to create recovery shard for {corrupted_shard.id}: {e}")
            return None
    
    def _repair_shard_links(self, shards: List[FileShard]) -> List[FileShard]:
        """
        Attempt to repair broken links between shards.
        
        Args:
            shards: List of shards with potentially broken links
            
        Returns:
            List of shards with repaired links
        """
        shard_map = {shard.id: shard for shard in shards}
        repaired_shards = []
        
        for shard in shards:
            # Create a copy to avoid modifying original
            repaired_shard = shard.clone()
            
            # Remove references to non-existent children
            valid_children = [
                child_id for child_id in repaired_shard.metadata.child_ids
                if child_id in shard_map
            ]
            repaired_shard.metadata.child_ids = valid_children
            
            # Remove references to non-existent parent
            if (repaired_shard.metadata.parent_id and 
                repaired_shard.metadata.parent_id not in shard_map):
                repaired_shard.metadata.parent_id = None
            
            # Remove references to non-existent next/previous shards
            if (repaired_shard.metadata.next_shard and 
                repaired_shard.metadata.next_shard not in shard_map):
                repaired_shard.metadata.next_shard = None
                
            if (repaired_shard.metadata.previous_shard and 
                repaired_shard.metadata.previous_shard not in shard_map):
                repaired_shard.metadata.previous_shard = None
            
            repaired_shards.append(repaired_shard)
        
        self.logger.info(f"Repaired links for {len(repaired_shards)} shards")
        return repaired_shards
    
    def validate_size_parameter(self, size: int) -> ValidationResult:
        """
        Validate size parameter for file operations.
        
        Args:
            size: Size parameter to validate
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        
        if size <= 0:
            errors.append(ValidationError(
                type=ErrorType.SIZE,
                message="Size parameter must be positive",
                location="size"
            ))
        elif size < 1024:  # Less than 1KB
            warnings.append("Size parameter is very small (< 1KB). This may create many small files.")
        elif size > 100 * 1024 * 1024:  # Greater than 100MB
            warnings.append("Size parameter is very large (> 100MB). This may not be optimal for LLM processing.")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_directory_path(self, path: str) -> ValidationResult:
        """
        Validate directory path for security and accessibility.
        
        Args:
            path: Directory path to validate
            
        Returns:
            ValidationResult with validation details
        """
        import os
        import pathlib
        
        errors = []
        warnings = []
        
        if not path:
            errors.append(ValidationError(
                type=ErrorType.PATH,
                message="Directory path cannot be empty",
                location="path"
            ))
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        try:
            # Resolve path to check for path traversal attempts
            resolved_path = pathlib.Path(path).resolve()
            
            # Check for path traversal attempts
            if ".." in str(resolved_path):
                errors.append(ValidationError(
                    type=ErrorType.PATH,
                    message="Path traversal detected in directory path",
                    location="path"
                ))
            
            # Check if path is accessible (if it exists)
            if resolved_path.exists():
                if not resolved_path.is_dir():
                    errors.append(ValidationError(
                        type=ErrorType.PATH,
                        message="Path exists but is not a directory",
                        location="path"
                    ))
                elif not os.access(resolved_path, os.R_OK | os.W_OK):
                    errors.append(ValidationError(
                        type=ErrorType.PATH,
                        message="Directory is not readable/writable",
                        location="path"
                    ))
            
        except (OSError, ValueError) as e:
            errors.append(ValidationError(
                type=ErrorType.PATH,
                message=f"Invalid directory path: {str(e)}",
                location="path"
            ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )