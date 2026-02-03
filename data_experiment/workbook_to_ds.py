#https://huggingface.co/datasets/internlm/Lean-Workbook


from config import generate_lean_example

import argparse
import json
import os
import random
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq


def estimate_row_size() -> int:
    """Estimate the size of a single row in bytes."""
    sample = generate_lean_example()
    # Estimate JSON size (includes keys, quotes, etc.)
    json_str = json.dumps(sample)
    return len(json_str.encode('utf-8'))


def generate_jsonl(output_path: str, target_size_mb: float, batch_size: int = 10000):
    """Generate a JSONL file up to the target size."""
    target_size_bytes = int(target_size_mb * 1024 * 1024)
    current_size = 0
    row_count = 0

    print(f"Generating JSONL file: {output_path}")
    print(f"Target size: {target_size_mb:.2f} MB ({target_size_bytes:,} bytes)")

    with open(output_path, 'w') as f:
        while current_size < target_size_bytes:
            # Generate a batch
            for _ in range(batch_size):
                example = generate_lean_example()
                line = json.dumps(example) + '\n'
                line_bytes = line.encode('utf-8')

                f.write(line)
                current_size += len(line_bytes)
                row_count += 1

                # Check if we've reached target size
                if current_size >= target_size_bytes:
                    break

            # Progress update
            size_mb = current_size / (1024 * 1024)
            print(f"Progress: {size_mb:.2f} MB written ({row_count:,} rows)", end='\r')

    final_size_mb = current_size / (1024 * 1024)
    print(f"\nCompleted! Final size: {final_size_mb:.2f} MB ({row_count:,} rows)")


def generate_parquet_by_rows(output_path: str, target_rows: int):
    """Generate a Parquet file with exactly the specified number of rows."""
    print(f"Generating Parquet file: {output_path}")
    print(f"Target rows: {target_rows:,}")

    # Define schema
    schema = pa.schema([
        ('user_prompt', pa.string()),
        ('system_prompt', pa.string()),
        ('ground_truth', pa.string())
    ])

    # Generate all examples
    batch_data = {
        'user_prompt': [],
        'system_prompt': [],
        'ground_truth': []
    }

    for i in range(target_rows):
        example = generate_lean_example()
        batch_data['user_prompt'].append(example['user_prompt'])
        batch_data['system_prompt'].append(example['system_prompt'])
        batch_data['ground_truth'].append(example['ground_truth'])

        if (i + 1) % 1000 == 0:
            print(f"Progress: {i + 1:,} / {target_rows:,} rows", end='\r')

    # Create Arrow table and write
    table = pa.table(batch_data, schema=schema)
    pq.write_table(table, output_path)

    final_size = os.path.getsize(output_path)
    final_size_kb = final_size / 1024
    print(f"\nCompleted! Final size: {final_size_kb:.2f} KB ({target_rows:,} rows)")


def generate_parquet(output_path: str, target_size_mb: float, batch_size: int = 100000):
    """Generate a Parquet file up to the target size."""
    target_size_bytes = int(target_size_mb * 1024 * 1024)

    print(f"Generating Parquet file: {output_path}")
    print(f"Target size: {target_size_mb:.2f} MB ({target_size_bytes:,} bytes)")

    # Define schema
    schema = pa.schema([
        ('user_prompt', pa.string()),
        ('system_prompt', pa.string()),
        ('ground_truth', pa.string())
    ])

    # Use ParquetWriter for streaming writes
    writer = None
    current_size = 0
    row_count = 0

    try:
        while current_size < target_size_bytes:
            # Generate a batch of examples
            batch_data = {
                'user_prompt': [],
                'system_prompt': [],
                'ground_truth': []
            }

            for _ in range(batch_size):
                example = generate_lean_example()
                batch_data['user_prompt'].append(example['user_prompt'])
                batch_data['system_prompt'].append(example['system_prompt'])
                batch_data['ground_truth'].append(example['ground_truth'])

            # Create Arrow table from batch
            table = pa.table(batch_data, schema=schema)

            # Initialize writer on first batch
            if writer is None:
                writer = pq.ParquetWriter(output_path, schema)

            # Write batch
            writer.write_table(table)
            row_count += len(batch_data['user_prompt'])

            # Update current size (approximate)
            if os.path.exists(output_path):
                current_size = os.path.getsize(output_path)

            # Progress update
            size_mb = current_size / (1024 * 1024)
            print(f"Progress: {size_mb:.2f} MB written ({row_count:,} rows)", end='\r')

            # Break if we've exceeded target size
            if current_size >= target_size_bytes:
                break

    finally:
        if writer:
            writer.close()

    final_size = os.path.getsize(output_path)
    final_size_mb = final_size / (1024 * 1024)
    print(f"\nCompleted! Final size: {final_size_mb:.2f} MB ({row_count:,} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data files for base conversion problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a file with exactly 1000 rows
  python parquet_generator.py data_1k.parquet --rows 1000

  # Generate a 2GB Parquet file
  python parquet_generator.py base_conversion_2gb.parquet --size 2048 --format parquet

  # Generate a 500MB file (auto-detect format from extension)
  python parquet_generator.py base_conversion_500mb.parquet --size 500
        """
    )

    parser.add_argument(
        'output',
            help='Output file path (e.g., base_conversion.parquet or base_conversion.jsonl)'
    )

    parser.add_argument(
        '--size',
        type=float,
        help='Target file size in MB (e.g., 2048 for 2GB, 500 for 500MB)'
    )

    parser.add_argument(
        '--rows',
        type=int,
        help='Target number of rows (alternative to --size)'
    )

    parser.add_argument(
        '--format',
        choices=['parquet', 'jsonl'],
        help='Output format (auto-detected from file extension if not specified)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for writing (default: 10000 for JSONL, 100000 for Parquet)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Determine format
    format_type = args.format
    if format_type is None:
        # Auto-detect from extension
        if args.output.endswith('.parquet'):
            format_type = 'parquet'
        elif args.output.endswith('.jsonl'):
            format_type = 'jsonl'
        else:
            parser.error("Cannot determine format from file extension. Please specify --format")

    # Validate that either --size or --rows is provided
    if args.size is None and args.rows is None:
        parser.error("Either --size or --rows must be specified")
    if args.size is not None and args.rows is not None:
        parser.error("Cannot specify both --size and --rows")

    # Generate file
    try:
        if args.rows is not None:
            # Row-based generation
            if args.rows <= 0:
                parser.error("Rows must be greater than 0")
            if format_type == 'jsonl':
                generate_jsonl_by_rows(args.output, args.rows)
            else:
                generate_parquet_by_rows(args.output, args.rows)
        else:
            # Size-based generation
            if args.size <= 0:
                parser.error("Size must be greater than 0")
            if args.size > 10240:
                print(f"Warning: Generating a {args.size}MB file. This may take a while...")

            batch_size = args.batch_size
            if batch_size is None:
                batch_size = 10000 if format_type == 'jsonl' else 100000

            if format_type == 'jsonl':
                generate_jsonl(args.output, args.size, batch_size)
            else:
                generate_parquet(args.output, args.size, batch_size)
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user")
        if os.path.exists(args.output):
            print(f"Partial file saved at: {args.output}")
        return 1
    except Exception as e:
        print(f"\n\nError: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
