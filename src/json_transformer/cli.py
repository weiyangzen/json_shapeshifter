"""Command-line interface for the JSON Transformer."""

import asyncio
import click
from pathlib import Path
from .json_transformer import JSONTransformer


@click.group()
@click.version_option(version="1.0.0")
def main():
    """JSON Transformer - Convert between compact JSON and LLM-readable files."""
    pass


@main.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--size', '-s', default=92160, help='Maximum file size in bytes (default: 90KB)')
@click.option('--output', '-o', default='./output', help='Output directory (default: ./output)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def unflatten(input_file: Path, size: int, output: str, verbose: bool):
    """Unflatten a JSON file into multiple LLM-readable files."""
    click.echo(f"Unflattening {input_file} to {output} with max size {size} bytes ({size/1024:.1f}KB)...")
    
    try:
        # Read input JSON file
        json_content = input_file.read_text(encoding='utf-8')
        
        # Create transformer and process
        transformer = JSONTransformer()
        result = asyncio.run(transformer.unflatten(json_content, size, output))
        
        if result.success:
            click.echo(f"✅ Successfully created {result.file_count} files in {result.output_directory}")
            click.echo(f"📊 Total size: {result.total_size} bytes")
        else:
            click.echo("❌ Unflatten operation failed:")
            for error in result.errors or []:
                click.echo(f"   • {error}")
                
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@main.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--output', '-o', help='Output JSON file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def flatten(input_dir: Path, output: str, verbose: bool):
    """Flatten a directory of LLM-readable files back to JSON."""
    click.echo(f"Flattening {input_dir} to JSON...")
    
    try:
        # Create transformer and process
        transformer = JSONTransformer()
        result = asyncio.run(transformer.flatten(str(input_dir)))
        
        if result.success:
            if output:
                # Write to file
                output_path = Path(output)
                output_path.write_text(result.json_string, encoding='utf-8')
                click.echo(f"✅ Successfully wrote JSON to {output_path}")
            else:
                # Print to stdout
                click.echo("✅ Reconstructed JSON:")
                click.echo(result.json_string)
        else:
            click.echo("❌ Flatten operation failed:")
            for error in result.errors or []:
                click.echo(f"   • {error}")
                
    except Exception as e:
        click.echo(f"❌ Error: {e}")


if __name__ == '__main__':
    main()