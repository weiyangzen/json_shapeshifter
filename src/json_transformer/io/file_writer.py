"""File writer utilities for JSON transformer output."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from ..models import FileShard, GlobalMetadata
from ..types import ProcessingError, ErrorType


class FileWriter:
    """
    File writer for creating LLM-readable shard files.
    
    Handles directory management, JSON serialization with proper formatting,
    and file naming conventions for the transformation output.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the file writer.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def write_shards(self, shards: List[FileShard], output_dir: str,
                    global_metadata: Optional[GlobalMetadata] = None) -> Dict[str, Any]:
        """
        Write all shards to files in the output directory.
        
        Args:
            shards: List of FileShard instances to write
            output_dir: Output directory path
            global_metadata: Optional global metadata to write
            
        Returns:
            Dictionary with write operation results
            
        Raises:
            ProcessingError: If writing fails
        """
        try:
            # Create output directory
            output_path = Path(output_dir)
            self._ensure_directory_exists(output_path)
            
            # Write results
            results = {
                "success": True,
                "output_directory": str(output_path.absolute()),
                "files_written": [],
                "total_size": 0,
                "errors": []
            }
            
            # Write global metadata if provided
            if global_metadata:
                metadata_file = self._write_global_metadata(output_path, global_metadata)
                if metadata_file:
                    results["files_written"].append(metadata_file)
            
            # Write each shard
            for shard in shards:
                try:
                    file_info = self._write_shard_file(output_path, shard)
                    results["files_written"].append(file_info)
                    results["total_size"] += file_info["size"]
                    
                except Exception as e:
                    error_msg = f"Failed to write shard {shard.id}: {str(e)}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)
                    results["success"] = False
            
            self.logger.info(f"Wrote {len(results['files_written'])} files to {output_dir}")
            return results
            
        except Exception as e:
            raise ProcessingError(
                f"Failed to write shards: {str(e)}",
                ErrorType.FILESYSTEM,
                context={"output_dir": output_dir, "shard_count": len(shards)}
            )
    
    def write_single_shard(self, shard: FileShard, output_dir: str,
                          filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Write a single shard to a file.
        
        Args:
            shard: FileShard to write
            output_dir: Output directory path
            filename: Optional custom filename
            
        Returns:
            Dictionary with file information
        """
        try:
            output_path = Path(output_dir)
            self._ensure_directory_exists(output_path)
            
            file_info = self._write_shard_file(output_path, shard, filename)
            
            self.logger.debug(f"Wrote shard {shard.id} to {file_info['path']}")
            return file_info
            
        except Exception as e:
            raise ProcessingError(
                f"Failed to write shard {shard.id}: {str(e)}",
                ErrorType.FILESYSTEM,
                context={"shard_id": shard.id, "output_dir": output_dir}
            )
    
    def _write_shard_file(self, output_path: Path, shard: FileShard,
                         custom_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Write a shard to a JSON file.
        
        Args:
            output_path: Output directory path
            shard: FileShard to write
            custom_filename: Optional custom filename
            
        Returns:
            Dictionary with file information
        """
        # Generate filename
        if custom_filename:
            filename = custom_filename
        else:
            filename = self._generate_filename(shard)
        
        file_path = output_path / filename
        
        # Create file schema
        file_schema = shard.to_file_schema()
        
        # Write JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(file_schema, f, indent=2, ensure_ascii=False)
        
        # Get file size
        file_size = file_path.stat().st_size
        
        return {
            "shard_id": shard.id,
            "filename": filename,
            "path": str(file_path.absolute()),
            "size": file_size,
            "data_type": shard.metadata.data_type.value
        }
    
    def _write_global_metadata(self, output_path: Path,
                              global_metadata: GlobalMetadata) -> Optional[Dict[str, Any]]:
        """
        Write global metadata to a file.
        
        Args:
            output_path: Output directory path
            global_metadata: GlobalMetadata to write
            
        Returns:
            Dictionary with file information or None if failed
        """
        try:
            metadata_file = output_path / "_global_metadata.json"
            
            metadata_dict = global_metadata.to_dict()
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            
            file_size = metadata_file.stat().st_size
            
            return {
                "shard_id": "_global_metadata",
                "filename": "_global_metadata.json",
                "path": str(metadata_file.absolute()),
                "size": file_size,
                "data_type": "metadata"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to write global metadata: {e}")
            return None
    
    def _generate_filename(self, shard: FileShard) -> str:
        """
        Generate filename for a shard.
        
        Args:
            shard: FileShard to generate filename for
            
        Returns:
            Generated filename
        """
        # Base filename from shard ID
        base_name = shard.id
        
        # Add data type suffix for clarity
        data_type_suffix = shard.metadata.data_type.value
        
        # Add path information if available
        if shard.metadata.original_path:
            path_suffix = "_".join(
                segment.replace("[", "").replace("]", "").replace(":", "_")
                for segment in shard.metadata.original_path[-2:]  # Last 2 path segments
                if segment
            )
            if path_suffix:
                base_name = f"{base_name}_{path_suffix}"
        
        return f"{base_name}_{data_type_suffix}.json"
    
    def _ensure_directory_exists(self, directory_path: Path) -> None:
        """
        Ensure that a directory exists, creating it if necessary.
        
        Args:
            directory_path: Path to directory
            
        Raises:
            ProcessingError: If directory creation fails
        """
        try:
            directory_path.mkdir(parents=True, exist_ok=True)
            
            # Check if directory is writable
            if not os.access(directory_path, os.W_OK):
                raise ProcessingError(
                    f"Directory {directory_path} is not writable",
                    ErrorType.FILESYSTEM
                )
                
        except OSError as e:
            raise ProcessingError(
                f"Failed to create directory {directory_path}: {str(e)}",
                ErrorType.FILESYSTEM
            )
    
    def create_index_file(self, output_dir: str, shards: List[FileShard],
                         global_metadata: Optional[GlobalMetadata] = None) -> Dict[str, Any]:
        """
        Create an index file listing all shards and their relationships.
        
        Args:
            output_dir: Output directory path
            shards: List of shards to index
            global_metadata: Optional global metadata
            
        Returns:
            Dictionary with index file information
        """
        try:
            output_path = Path(output_dir)
            index_file = output_path / "_index.json"
            
            # Create index structure
            index_data = {
                "version": "1.0.0",
                "created_at": global_metadata.created_at.isoformat() if global_metadata else None,
                "total_shards": len(shards),
                "root_shard": global_metadata.root_shard if global_metadata else None,
                "data_structure": global_metadata.data_structure.value if global_metadata else None,
                "shards": []
            }
            
            # Add shard information
            for shard in shards:
                shard_info = {
                    "shard_id": shard.id,
                    "filename": self._generate_filename(shard),
                    "data_type": shard.metadata.data_type.value,
                    "original_path": shard.metadata.original_path,
                    "parent_id": shard.metadata.parent_id,
                    "child_ids": shard.metadata.child_ids,
                    "next_shard": shard.metadata.next_shard,
                    "previous_shard": shard.metadata.previous_shard,
                    "size": shard.size
                }
                index_data["shards"].append(shard_info)
            
            # Write index file
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            
            file_size = index_file.stat().st_size
            
            self.logger.info(f"Created index file with {len(shards)} shards")
            
            return {
                "filename": "_index.json",
                "path": str(index_file.absolute()),
                "size": file_size,
                "shard_count": len(shards)
            }
            
        except Exception as e:
            raise ProcessingError(
                f"Failed to create index file: {str(e)}",
                ErrorType.FILESYSTEM,
                context={"output_dir": output_dir}
            )
    
    def validate_output_directory(self, output_dir: str) -> Dict[str, Any]:
        """
        Validate that the output directory is suitable for writing.
        
        Args:
            output_dir: Directory path to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "path": output_dir,
            "exists": False,
            "is_writable": False,
            "is_empty": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            output_path = Path(output_dir)
            
            # Check if path exists
            if output_path.exists():
                validation_result["exists"] = True
                
                if not output_path.is_dir():
                    validation_result["errors"].append("Path exists but is not a directory")
                    validation_result["is_valid"] = False
                    return validation_result
                
                # Check if directory is empty
                if any(output_path.iterdir()):
                    validation_result["is_empty"] = False
                    validation_result["warnings"].append("Directory is not empty - files may be overwritten")
                
                # Check write permissions
                if os.access(output_path, os.W_OK):
                    validation_result["is_writable"] = True
                else:
                    validation_result["errors"].append("Directory is not writable")
                    validation_result["is_valid"] = False
            else:
                # Check if parent directory is writable
                parent_path = output_path.parent
                if parent_path.exists() and os.access(parent_path, os.W_OK):
                    validation_result["is_writable"] = True
                else:
                    validation_result["errors"].append("Cannot create directory - parent is not writable")
                    validation_result["is_valid"] = False
        
        except Exception as e:
            validation_result["errors"].append(f"Validation failed: {str(e)}")
            validation_result["is_valid"] = False
        
        return validation_result
    
    def cleanup_output_directory(self, output_dir: str, pattern: str = "*.json") -> Dict[str, Any]:
        """
        Clean up files in the output directory.
        
        Args:
            output_dir: Directory to clean up
            pattern: File pattern to match (default: *.json)
            
        Returns:
            Dictionary with cleanup results
        """
        try:
            output_path = Path(output_dir)
            
            if not output_path.exists():
                return {
                    "success": True,
                    "files_removed": 0,
                    "message": "Directory does not exist"
                }
            
            # Find matching files
            matching_files = list(output_path.glob(pattern))
            files_removed = 0
            
            for file_path in matching_files:
                try:
                    file_path.unlink()
                    files_removed += 1
                except Exception as e:
                    self.logger.warning(f"Failed to remove {file_path}: {e}")
            
            self.logger.info(f"Cleaned up {files_removed} files from {output_dir}")
            
            return {
                "success": True,
                "files_removed": files_removed,
                "message": f"Removed {files_removed} files"
            }
            
        except Exception as e:
            return {
                "success": False,
                "files_removed": 0,
                "message": f"Cleanup failed: {str(e)}"
            }
    
    def get_output_statistics(self, output_dir: str) -> Dict[str, Any]:
        """
        Get statistics about the output directory.
        
        Args:
            output_dir: Directory to analyze
            
        Returns:
            Dictionary with statistics
        """
        try:
            output_path = Path(output_dir)
            
            if not output_path.exists():
                return {
                    "exists": False,
                    "total_files": 0,
                    "total_size": 0,
                    "file_types": {}
                }
            
            # Analyze files
            json_files = list(output_path.glob("*.json"))
            total_size = 0
            file_types = {}
            
            for file_path in json_files:
                file_size = file_path.stat().st_size
                total_size += file_size
                
                # Categorize file types
                if file_path.name.startswith("_"):
                    file_type = "metadata"
                elif "_dict.json" in file_path.name:
                    file_type = "dict_shard"
                elif "_list.json" in file_path.name:
                    file_type = "list_shard"
                elif "_primitive.json" in file_path.name:
                    file_type = "primitive_shard"
                else:
                    file_type = "other"
                
                file_types[file_type] = file_types.get(file_type, 0) + 1
            
            return {
                "exists": True,
                "total_files": len(json_files),
                "total_size": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "file_types": file_types,
                "average_file_size": total_size / len(json_files) if json_files else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get output statistics: {e}")
            return {
                "exists": False,
                "error": str(e)
            }