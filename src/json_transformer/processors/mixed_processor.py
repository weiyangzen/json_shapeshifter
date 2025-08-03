"""Mixed structure processor for handling complex JSON with both dicts and lists."""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from ..types import DataProcessorInterface, ProcessingResult, DataStructure, DataType
from ..models import FileShard, ShardMetadata, GlobalMetadata
from ..engines.file_sharding_engine import FileShardingEngine
from ..utils.size_calculator import SizeCalculator
from .dict_processor import DictProcessor
from .list_processor import ListProcessor


class MixedProcessor(DataProcessorInterface):
    """
    Processor for mixed data structures containing both dictionaries and lists.
    
    Routes different data types to appropriate processors while maintaining
    overall structure integrity and creating integration between different
    processing strategies.
    """
    
    def __init__(self, sharding_engine: Optional[FileShardingEngine] = None,
                 size_calculator: Optional[SizeCalculator] = None,
                 dict_processor: Optional[DictProcessor] = None,
                 list_processor: Optional[ListProcessor] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the mixed structure processor.
        
        Args:
            sharding_engine: Optional FileShardingEngine instance
            size_calculator: Optional SizeCalculator instance
            dict_processor: Optional DictProcessor instance
            list_processor: Optional ListProcessor instance
            logger: Optional logger instance
        """
        self.sharding_engine = sharding_engine or FileShardingEngine()
        self.size_calculator = size_calculator or SizeCalculator()
        self.dict_processor = dict_processor or DictProcessor(
            self.sharding_engine, self.size_calculator
        )
        self.list_processor = list_processor or ListProcessor(
            self.sharding_engine, self.size_calculator
        )
        self.logger = logger or logging.getLogger(__name__)
    
    def process(self, data: Any, max_size: int) -> ProcessingResult:
        """
        Process mixed structure data into shards.
        
        Args:
            data: Mixed structure data to process
            max_size: Maximum size per shard in bytes
            
        Returns:
            ProcessingResult with shards and metadata
            
        Raises:
            ValueError: If data processing fails
        """
        self.logger.info(f"Processing mixed structure data, max_size={max_size}")
        
        # Calculate original size
        original_size = self.size_calculator.calculate_json_size(data)
        
        # Analyze the mixed structure
        structure_analysis = self._analyze_mixed_structure(data)
        
        # Process based on root type
        if isinstance(data, dict):
            all_shards = self._process_mixed_dict(data, max_size, structure_analysis)
        elif isinstance(data, list):
            all_shards = self._process_mixed_list(data, max_size, structure_analysis)
        else:
            raise ValueError(f"Unsupported root data type: {type(data).__name__}")
        
        # Create global metadata
        global_metadata = GlobalMetadata(
            version="1.0.0",
            original_size=original_size,
            shard_count=len(all_shards),
            root_shard=all_shards[0].id if all_shards else "empty",
            created_at=datetime.now(),
            data_structure=DataStructure.MIXED
        )
        
        self.logger.info(f"Created {len(all_shards)} shards from mixed structure processing")
        
        return ProcessingResult(
            shards=all_shards,
            root_shard_id=all_shards[0].id if all_shards else "empty",
            metadata=global_metadata
        )
    
    def _analyze_mixed_structure(self, data: Any) -> Dict[str, Any]:
        """
        Analyze the mixed structure to understand its composition.
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary with structure analysis
        """
        analysis = {
            "root_type": type(data).__name__,
            "dict_count": 0,
            "list_count": 0,
            "primitive_count": 0,
            "max_depth": 0,
            "complexity_score": 0,
            "structure_map": {}
        }
        
        # Recursively analyze structure
        self._analyze_structure_recursive(data, analysis, [], 0)
        
        # Calculate complexity score
        analysis["complexity_score"] = self._calculate_complexity_score(analysis)
        
        return analysis
    
    def _analyze_structure_recursive(self, data: Any, analysis: Dict[str, Any], 
                                   path: List[str], depth: int) -> None:
        """
        Recursively analyze structure composition.
        
        Args:
            data: Current data to analyze
            analysis: Analysis dictionary to update
            path: Current path in the structure
            depth: Current depth level
        """
        analysis["max_depth"] = max(analysis["max_depth"], depth)
        path_key = ".".join(path) if path else "root"
        
        if isinstance(data, dict):
            analysis["dict_count"] += 1
            analysis["structure_map"][path_key] = {
                "type": "dict",
                "size": len(data),
                "depth": depth
            }
            
            for key, value in data.items():
                self._analyze_structure_recursive(
                    value, analysis, path + [str(key)], depth + 1
                )
        
        elif isinstance(data, list):
            analysis["list_count"] += 1
            analysis["structure_map"][path_key] = {
                "type": "list",
                "size": len(data),
                "depth": depth
            }
            
            # Sample first few items to understand list structure
            sample_size = min(5, len(data))
            for i in range(sample_size):
                self._analyze_structure_recursive(
                    data[i], analysis, path + [f"[{i}]"], depth + 1
                )
        
        else:
            analysis["primitive_count"] += 1
            analysis["structure_map"][path_key] = {
                "type": "primitive",
                "value_type": type(data).__name__,
                "depth": depth
            }
    
    def _calculate_complexity_score(self, analysis: Dict[str, Any]) -> float:
        """
        Calculate complexity score for the mixed structure.
        
        Args:
            analysis: Structure analysis
            
        Returns:
            Complexity score (0.0 to 10.0)
        """
        score = 0.0
        
        # Base score on structure diversity
        structure_types = len(set(
            item["type"] for item in analysis["structure_map"].values()
        ))
        score += structure_types * 0.5
        
        # Add score for depth
        score += analysis["max_depth"] * 0.3
        
        # Add score for total elements
        total_elements = (analysis["dict_count"] + 
                         analysis["list_count"] + 
                         analysis["primitive_count"])
        if total_elements > 100:
            score += 1.0
        elif total_elements > 1000:
            score += 2.0
        
        # Add score for mixed types at same level
        depth_types = {}
        for item in analysis["structure_map"].values():
            depth = item["depth"]
            if depth not in depth_types:
                depth_types[depth] = set()
            depth_types[depth].add(item["type"])
        
        for types_at_depth in depth_types.values():
            if len(types_at_depth) > 1:
                score += 0.5
        
        return min(score, 10.0)  # Cap at 10.0
    
    def _process_mixed_dict(self, data: Dict[str, Any], max_size: int, 
                           analysis: Dict[str, Any]) -> List[FileShard]:
        """
        Process mixed dictionary structure.
        
        Args:
            data: Dictionary data to process
            max_size: Maximum size per shard
            analysis: Pre-computed structure analysis
            
        Returns:
            List of FileShard instances
        """
        all_shards = []
        
        # Separate different types of values
        dict_values = {}
        list_values = {}
        primitive_values = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                dict_values[key] = value
            elif isinstance(value, list):
                list_values[key] = value
            else:
                primitive_values[key] = value
        
        # Create root shard with primitive values and references
        root_shard_data = primitive_values.copy()
        root_shard_id = self.sharding_engine.generate_shard_id()
        child_ids = []
        
        # Process dictionary values using dict processor
        for key, dict_value in dict_values.items():
            try:
                dict_result = self.dict_processor.process(dict_value, max_size)
                all_shards.extend(dict_result.shards)
                
                # Update parent relationships
                for shard in dict_result.shards:
                    if shard.id == dict_result.root_shard_id:
                        shard.metadata.parent_id = root_shard_id
                        child_ids.append(shard.id)
                        
                        # Add reference in root shard
                        root_shard_data[key] = {
                            "_shard_ref": shard.id,
                            "_shard_type": "dict",
                            "_processor": "dict_processor",
                            "_original_path": [key]
                        }
                        break
                
            except Exception as e:
                self.logger.error(f"Failed to process dict value '{key}': {e}")
                # Include as primitive with error marker
                root_shard_data[key] = {
                    "_error": f"Processing failed: {str(e)}",
                    "_original_type": "dict"
                }
        
        # Process list values using list processor
        for key, list_value in list_values.items():
            try:
                list_result = self.list_processor.process(list_value, max_size)
                all_shards.extend(list_result.shards)
                
                # Update parent relationships
                for shard in list_result.shards:
                    if shard.id == list_result.root_shard_id:
                        shard.metadata.parent_id = root_shard_id
                        child_ids.append(shard.id)
                        
                        # Add reference in root shard
                        root_shard_data[key] = {
                            "_shard_ref": shard.id,
                            "_shard_type": "list",
                            "_processor": "list_processor",
                            "_original_path": [key]
                        }
                        break
                
            except Exception as e:
                self.logger.error(f"Failed to process list value '{key}': {e}")
                # Include as primitive with error marker
                root_shard_data[key] = {
                    "_error": f"Processing failed: {str(e)}",
                    "_original_type": "list"
                }
        
        # Check if root shard data exceeds size limit
        if self.sharding_engine.calculate_size(root_shard_data) > max_size:
            # Split root shard data
            root_chunks = self.sharding_engine.split_data_by_size(
                root_shard_data, max_size, []
            )
            
            # Create shards from chunks
            root_shards = self.sharding_engine.create_shards_from_chunks(
                root_chunks, None
            )
            
            # Update the root shard ID to the first chunk
            if root_shards:
                root_shard_id = root_shards[0].id
                all_shards = root_shards + all_shards
        else:
            # Create single root shard
            root_metadata = ShardMetadata(
                shard_id=root_shard_id,
                parent_id=None,
                child_ids=child_ids,
                data_type=DataType.DICT,
                original_path=[]
            )
            
            root_shard = self.sharding_engine.create_shard(
                root_shard_data, root_shard_id, root_metadata
            )
            all_shards.insert(0, root_shard)
        
        return all_shards
    
    def _process_mixed_list(self, data: List[Any], max_size: int, 
                           analysis: Dict[str, Any]) -> List[FileShard]:
        """
        Process mixed list structure.
        
        Args:
            data: List data to process
            max_size: Maximum size per shard
            analysis: Pre-computed structure analysis
            
        Returns:
            List of FileShard instances
        """
        all_shards = []
        current_chunk = []
        current_size = 0
        chunk_start_index = 0
        
        for i, item in enumerate(data):
            # Determine processing strategy for this item
            if isinstance(item, dict):
                # Process as dictionary if it's complex enough
                if self._should_process_as_dict(item, max_size):
                    # Save current chunk if it has items
                    if current_chunk:
                        chunk_shards = self._create_list_chunk_shards(
                            current_chunk, chunk_start_index, max_size
                        )
                        all_shards.extend(chunk_shards)
                        current_chunk = []
                        current_size = 0
                    
                    # Process dictionary item
                    try:
                        dict_result = self.dict_processor.process(item, max_size)
                        
                        # Create wrapper shard for the dictionary
                        wrapper_data = {
                            "_list_item_index": i,
                            "_shard_ref": dict_result.root_shard_id,
                            "_shard_type": "dict",
                            "_processor": "dict_processor"
                        }
                        
                        wrapper_shard = self._create_wrapper_shard(
                            wrapper_data, i, DataType.DICT
                        )
                        
                        all_shards.append(wrapper_shard)
                        all_shards.extend(dict_result.shards)
                        
                        # Update parent relationships
                        for shard in dict_result.shards:
                            if shard.id == dict_result.root_shard_id:
                                shard.metadata.parent_id = wrapper_shard.id
                                wrapper_shard.metadata.child_ids.append(shard.id)
                                break
                    
                    except Exception as e:
                        self.logger.error(f"Failed to process dict item at index {i}: {e}")
                        # Add as regular item with error marker
                        error_item = {
                            "_error": f"Processing failed: {str(e)}",
                            "_original_index": i,
                            "_original_type": "dict"
                        }
                        current_chunk.append(error_item)
                        current_size += self.size_calculator.calculate_json_size(error_item)
                    
                    chunk_start_index = i + 1
                    continue
            
            elif isinstance(item, list):
                # Process as list if it's complex enough
                if self._should_process_as_list(item, max_size):
                    # Save current chunk if it has items
                    if current_chunk:
                        chunk_shards = self._create_list_chunk_shards(
                            current_chunk, chunk_start_index, max_size
                        )
                        all_shards.extend(chunk_shards)
                        current_chunk = []
                        current_size = 0
                    
                    # Process list item
                    try:
                        list_result = self.list_processor.process(item, max_size)
                        
                        # Create wrapper shard for the list
                        wrapper_data = {
                            "_list_item_index": i,
                            "_shard_ref": list_result.root_shard_id,
                            "_shard_type": "list",
                            "_processor": "list_processor"
                        }
                        
                        wrapper_shard = self._create_wrapper_shard(
                            wrapper_data, i, DataType.LIST
                        )
                        
                        all_shards.append(wrapper_shard)
                        all_shards.extend(list_result.shards)
                        
                        # Update parent relationships
                        for shard in list_result.shards:
                            if shard.id == list_result.root_shard_id:
                                shard.metadata.parent_id = wrapper_shard.id
                                wrapper_shard.metadata.child_ids.append(shard.id)
                                break
                    
                    except Exception as e:
                        self.logger.error(f"Failed to process list item at index {i}: {e}")
                        # Add as regular item with error marker
                        error_item = {
                            "_error": f"Processing failed: {str(e)}",
                            "_original_index": i,
                            "_original_type": "list"
                        }
                        current_chunk.append(error_item)
                        current_size += self.size_calculator.calculate_json_size(error_item)
                    
                    chunk_start_index = i + 1
                    continue
            
            # Handle as regular list item
            try:
                item_size = self.size_calculator.calculate_json_size(item)
            except ValueError:
                # Skip non-serializable items
                self.logger.warning(f"Skipping non-serializable item at index {i}")
                continue
            
            # Check if adding this item would exceed size limit
            if self.sharding_engine.should_create_new_shard(current_size, item_size, max_size):
                # Save current chunk
                if current_chunk:
                    chunk_shards = self._create_list_chunk_shards(
                        current_chunk, chunk_start_index, max_size
                    )
                    all_shards.extend(chunk_shards)
                
                # Start new chunk
                current_chunk = [item]
                current_size = item_size
                chunk_start_index = i
            else:
                # Add to current chunk
                current_chunk.append(item)
                current_size += item_size
        
        # Handle final chunk
        if current_chunk:
            chunk_shards = self._create_list_chunk_shards(
                current_chunk, chunk_start_index, max_size
            )
            all_shards.extend(chunk_shards)
        
        # Set up sequential linking for list shards
        list_shards = [s for s in all_shards if s.metadata.data_type == DataType.LIST]
        self._setup_list_sequential_linking(list_shards)
        
        return all_shards
    
    def _should_process_as_dict(self, item: Dict[str, Any], max_size: int) -> bool:
        """
        Determine if a dictionary item should be processed separately.
        
        Args:
            item: Dictionary item to evaluate
            max_size: Maximum size constraint
            
        Returns:
            True if item should be processed as separate dictionary
        """
        try:
            item_size = self.size_calculator.calculate_json_size(item)
            
            # Process separately if:
            # 1. Item is large (> 30% of max size)
            # 2. Item has nested structures
            # 3. Item has many keys (> 10)
            
            if item_size > max_size * 0.3:
                return True
            
            if len(item) > 10:
                return True
            
            # Check for nested structures
            for value in item.values():
                if isinstance(value, (dict, list)):
                    return True
            
            return False
            
        except ValueError:
            return False  # Can't process non-serializable items separately
    
    def _should_process_as_list(self, item: List[Any], max_size: int) -> bool:
        """
        Determine if a list item should be processed separately.
        
        Args:
            item: List item to evaluate
            max_size: Maximum size constraint
            
        Returns:
            True if item should be processed as separate list
        """
        try:
            item_size = self.size_calculator.calculate_json_size(item)
            
            # Process separately if:
            # 1. Item is large (> 30% of max size)
            # 2. Item has many elements (> 20)
            # 3. Item has nested structures
            
            if item_size > max_size * 0.3:
                return True
            
            if len(item) > 20:
                return True
            
            # Check for nested structures
            for element in item:
                if isinstance(element, (dict, list)):
                    return True
            
            return False
            
        except ValueError:
            return False  # Can't process non-serializable items separately
    
    def _create_wrapper_shard(self, data: Dict[str, Any], index: int, 
                             data_type: DataType) -> FileShard:
        """
        Create a wrapper shard for a processed item.
        
        Args:
            data: Wrapper data
            index: Original index in the list
            data_type: Type of the wrapped data
            
        Returns:
            FileShard instance
        """
        shard_id = self.sharding_engine.generate_shard_id()
        
        metadata = ShardMetadata(
            shard_id=shard_id,
            parent_id=None,
            child_ids=[],
            data_type=data_type,
            original_path=[f"[{index}]"]
        )
        
        return self.sharding_engine.create_shard(data, shard_id, metadata)
    
    def _create_list_chunk_shards(self, chunk: List[Any], start_index: int, 
                                 max_size: int) -> List[FileShard]:
        """
        Create shards from a list chunk.
        
        Args:
            chunk: List chunk to create shards from
            start_index: Starting index of this chunk
            max_size: Maximum size per shard
            
        Returns:
            List of FileShard instances
        """
        # Use list processor's chunk creation logic
        return self.list_processor._create_chunk_shards(chunk, start_index, max_size)
    
    def _setup_list_sequential_linking(self, shards: List[FileShard]) -> None:
        """
        Set up sequential linking for list shards.
        
        Args:
            shards: List of shards to link
        """
        for i in range(len(shards)):
            if i > 0:
                shards[i].metadata.previous_shard = shards[i-1].id
            if i < len(shards) - 1:
                shards[i].metadata.next_shard = shards[i+1].id
    
    def get_mixed_structure_statistics(self, shards: List[FileShard]) -> Dict[str, Any]:
        """
        Get statistics about mixed structure processing results.
        
        Args:
            shards: List of processed shards
            
        Returns:
            Dictionary with statistics
        """
        if not shards:
            return {
                "total_shards": 0,
                "processor_distribution": {},
                "data_type_distribution": {},
                "complexity_metrics": {}
            }
        
        # Count shards by processor type
        processor_counts = {"dict_processor": 0, "list_processor": 0, "mixed_processor": 0}
        data_type_counts = {}
        
        for shard in shards:
            # Count data types
            data_type = shard.metadata.data_type.value
            data_type_counts[data_type] = data_type_counts.get(data_type, 0) + 1
            
            # Try to determine processor type from shard data
            if isinstance(shard.data, dict):
                if "_processor" in shard.data:
                    processor = shard.data["_processor"]
                    if processor in processor_counts:
                        processor_counts[processor] += 1
                else:
                    processor_counts["mixed_processor"] += 1
            else:
                processor_counts["mixed_processor"] += 1
        
        # Calculate complexity metrics
        total_size = sum(shard.size for shard in shards)
        avg_size = total_size / len(shards) if shards else 0
        
        # Count hierarchical relationships
        parent_child_pairs = 0
        sequential_links = 0
        
        for shard in shards:
            if shard.metadata.child_ids:
                parent_child_pairs += len(shard.metadata.child_ids)
            if shard.metadata.next_shard:
                sequential_links += 1
        
        return {
            "total_shards": len(shards),
            "processor_distribution": processor_counts,
            "data_type_distribution": data_type_counts,
            "complexity_metrics": {
                "total_size": total_size,
                "average_shard_size": avg_size,
                "parent_child_relationships": parent_child_pairs,
                "sequential_links": sequential_links,
                "structure_diversity": len(data_type_counts)
            }
        }
    
    def validate_mixed_structure_integrity(self, shards: List[FileShard]) -> Dict[str, Any]:
        """
        Validate the integrity of mixed structure processing.
        
        Args:
            shards: List of shards to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        if not shards:
            validation_result["warnings"].append("No shards to validate")
            return validation_result
        
        shard_map = {shard.id: shard for shard in shards}
        
        # Validate references
        for shard in shards:
            if isinstance(shard.data, dict):
                for key, value in shard.data.items():
                    if isinstance(value, dict) and "_shard_ref" in value:
                        ref_id = value["_shard_ref"]
                        if ref_id not in shard_map:
                            validation_result["errors"].append(
                                f"Broken reference: {shard.id} -> {ref_id}"
                            )
                            validation_result["is_valid"] = False
        
        # Validate hierarchical relationships
        for shard in shards:
            for child_id in shard.metadata.child_ids:
                if child_id not in shard_map:
                    validation_result["errors"].append(
                        f"Missing child shard: {shard.id} -> {child_id}"
                    )
                    validation_result["is_valid"] = False
                else:
                    child_shard = shard_map[child_id]
                    if child_shard.metadata.parent_id != shard.id:
                        validation_result["errors"].append(
                            f"Broken parent-child link: {shard.id} <-> {child_id}"
                        )
                        validation_result["is_valid"] = False
        
        # Validate sequential links
        for shard in shards:
            if shard.metadata.next_shard:
                next_id = shard.metadata.next_shard
                if next_id not in shard_map:
                    validation_result["errors"].append(
                        f"Missing next shard: {shard.id} -> {next_id}"
                    )
                    validation_result["is_valid"] = False
                else:
                    next_shard = shard_map[next_id]
                    if next_shard.metadata.previous_shard != shard.id:
                        validation_result["errors"].append(
                            f"Broken sequential link: {shard.id} <-> {next_id}"
                        )
                        validation_result["is_valid"] = False
        
        return validation_result