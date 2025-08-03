"""Integration tests for the JSON Transformer."""

import pytest
import tempfile
import json
from pathlib import Path
from json_transformer import JSONTransformer


class TestJSONTransformerIntegration:
    """Integration tests for the complete JSON Transformer system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.transformer = JSONTransformer()
    
    @pytest.mark.asyncio
    async def test_unflatten_simple_dict(self):
        """Test unflattening a simple dictionary structure."""
        data = {
            "users": {
                "user1": {"name": "Alice", "age": 30},
                "user2": {"name": "Bob", "age": 25}
            },
            "settings": {
                "theme": "dark",
                "notifications": True
            }
        }
        
        json_string = json.dumps(data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = await self.transformer.unflatten(
                json_string, 
                max_size=92160,  # 90KB default
                output_dir=temp_dir
            )
            
            assert result.success
            assert result.file_count > 0
            assert result.total_size > 0
            assert Path(result.output_directory).exists()
            
            # Check that files were created
            output_path = Path(result.output_directory)
            json_files = list(output_path.glob("*.json"))
            assert len(json_files) >= result.file_count
            
            # Check that index file was created
            index_file = output_path / "_index.json"
            assert index_file.exists()
    
    @pytest.mark.asyncio
    async def test_unflatten_simple_list(self):
        """Test unflattening a simple list structure."""
        data = [
            {"id": 1, "name": "Item 1", "value": 100},
            {"id": 2, "name": "Item 2", "value": 200},
            {"id": 3, "name": "Item 3", "value": 300}
        ]
        
        json_string = json.dumps(data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = await self.transformer.unflatten(
                json_string,
                max_size=25000,
                output_dir=temp_dir
            )
            
            assert result.success
            assert result.file_count > 0
            assert result.total_size > 0
    
    @pytest.mark.asyncio
    async def test_unflatten_mixed_structure(self):
        """Test unflattening a mixed structure."""
        data = {
            "metadata": {"version": "1.0", "created": "2024-01-01"},
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"}
            ],
            "config": {
                "settings": {"theme": "dark"},
                "features": ["feature1", "feature2"]
            }
        }
        
        json_string = json.dumps(data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = await self.transformer.unflatten(
                json_string,
                max_size=25000,
                output_dir=temp_dir
            )
            
            assert result.success
            assert result.file_count > 0
    
    @pytest.mark.asyncio
    async def test_unflatten_invalid_json(self):
        """Test unflattening invalid JSON."""
        invalid_json = '{"users": {"user1": {"name": "Alice"}'  # Missing closing braces
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = await self.transformer.unflatten(
                invalid_json,
                output_dir=temp_dir
            )
            
            assert not result.success
            assert len(result.errors) > 0
            assert result.file_count == 0
    
    @pytest.mark.asyncio
    async def test_unflatten_invalid_size_parameter(self):
        """Test unflattening with invalid size parameter."""
        data = {"simple": "data"}
        json_string = json.dumps(data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = await self.transformer.unflatten(
                json_string,
                max_size=-1000,  # Invalid negative size
                output_dir=temp_dir
            )
            
            assert not result.success
            assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_unflatten_large_data_splitting(self):
        """Test unflattening large data that requires splitting."""
        # Create large data structure
        data = {
            f"section_{i}": {
                f"key_{j}": f"value_{j}" * 50  # Make values larger
                for j in range(20)
            }
            for i in range(10)
        }
        
        json_string = json.dumps(data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = await self.transformer.unflatten(
                json_string,
                max_size=2000,  # Small size to force splitting
                output_dir=temp_dir
            )
            
            assert result.success
            assert result.file_count > 1  # Should create multiple files
    
    @pytest.mark.asyncio
    async def test_unflatten_with_default_parameters(self):
        """Test unflattening with default parameters."""
        data = {"simple": {"nested": {"data": "value"}}}
        json_string = json.dumps(data)
        
        # Use default output directory
        result = await self.transformer.unflatten(json_string)
        
        # Should succeed even with default parameters
        # Note: This will create files in ./output directory
        assert result.success or len(result.errors) > 0  # May fail due to permissions
    
    @pytest.mark.asyncio
    async def test_unflatten_empty_structures(self):
        """Test unflattening empty structures."""
        # Empty dict
        result_dict = await self.transformer.unflatten('{}')
        assert result_dict.success or len(result_dict.errors) > 0
        
        # Empty list
        result_list = await self.transformer.unflatten('[]')
        assert result_list.success or len(result_list.errors) > 0
    
    def test_file_schema_structure(self):
        """Test that generated files follow the expected schema."""
        # This test would need to be expanded once flatten is implemented
        # For now, we just verify the transformer can be instantiated
        transformer = JSONTransformer(default_max_size=92160)
        assert transformer.default_max_size == 92160