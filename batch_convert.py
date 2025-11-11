#!/usr/bin/env python3
"""
Batch PDF to SVG Converter
Process multiple PDF files in parallel
"""

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from pdf_to_svg import PDFtoSVGConverter
import sys


def convert_single_pdf(pdf_path: str, output_dir: str, converter_kwargs: dict) -> dict:
    """Convert a single PDF file"""
    try:
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_svg = output_dir / pdf_path.with_suffix('.svg').name

        converter = PDFtoSVGConverter(**converter_kwargs)
        result_path = converter.convert(str(pdf_path), str(output_svg))

        # Get file sizes
        pdf_size = pdf_path.stat().st_size
        svg_size = Path(result_path).stat().st_size

        return {
            'success': True,
            'input': str(pdf_path),
            'output': result_path,
            'pdf_size': pdf_size,
            'svg_size': svg_size
        }

    except Exception as e:
        return {
            'success': False,
            'input': str(pdf_path),
            'error': str(e)
        }


def find_pdf_files(input_paths: list) -> list:
    """Find all PDF files from input paths (files or directories)"""
    pdf_files = []

    for path_str in input_paths:
        path = Path(path_str)

        if not path.exists():
            print(f"Warning: Path not found: {path}", file=sys.stderr)
            continue

        if path.is_file() and path.suffix.lower() == '.pdf':
            pdf_files.append(path)
        elif path.is_dir():
            # Recursively find all PDFs in directory
            pdf_files.extend(path.rglob('*.pdf'))
            pdf_files.extend(path.rglob('*.PDF'))

    return sorted(set(pdf_files))


def main():
    parser = argparse.ArgumentParser(
        description='Batch convert PDF technical drawings to SVG with arc reconstruction'
    )
    parser.add_argument('inputs', nargs='+',
                       help='Input PDF files or directories (will search recursively)')
    parser.add_argument('-o', '--output-dir', default='output',
                       help='Output directory for SVG files (default: output/)')
    parser.add_argument('-j', '--jobs', type=int, default=4,
                       help='Number of parallel jobs (default: 4)')
    parser.add_argument('--no-arc-detection', action='store_true',
                       help='Disable arc detection (output polylines only)')
    parser.add_argument('--angle-tolerance', type=float, default=8.0,
                       help='Angle tolerance for arc detection (degrees, default: 8.0)')
    parser.add_argument('--radius-tolerance', type=float, default=0.03,
                       help='Radius tolerance for arc detection (fraction, default: 0.03)')
    parser.add_argument('--min-arc-points', type=int, default=4,
                       help='Minimum points to consider as arc (default: 4)')

    args = parser.parse_args()

    # Find all PDF files
    pdf_files = find_pdf_files(args.inputs)

    if not pdf_files:
        print("Error: No PDF files found", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF file(s) to convert")
    print(f"Output directory: {args.output_dir}")
    print(f"Parallel jobs: {args.jobs}")
    print()

    # Converter parameters
    converter_kwargs = {
        'arc_detection': not args.no_arc_detection,
        'angle_tolerance': args.angle_tolerance,
        'radius_tolerance': args.radius_tolerance,
        'min_arc_points': args.min_arc_points
    }

    # Process files
    results = []
    successful = 0
    failed = 0

    if args.jobs == 1:
        # Sequential processing
        for pdf_file in pdf_files:
            print(f"Converting: {pdf_file.name}...", end=' ')
            sys.stdout.flush()

            result = convert_single_pdf(str(pdf_file), args.output_dir, converter_kwargs)
            results.append(result)

            if result['success']:
                print(f"OK ({result['svg_size']:,} bytes)")
                successful += 1
            else:
                print(f"FAILED: {result['error']}")
                failed += 1
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            # Submit all jobs
            future_to_pdf = {
                executor.submit(convert_single_pdf, str(pdf_file), args.output_dir, converter_kwargs): pdf_file
                for pdf_file in pdf_files
            }

            # Process results as they complete
            for future in as_completed(future_to_pdf):
                pdf_file = future_to_pdf[future]
                print(f"Converting: {pdf_file.name}...", end=' ')
                sys.stdout.flush()

                result = future.result()
                results.append(result)

                if result['success']:
                    print(f"OK ({result['svg_size']:,} bytes)")
                    successful += 1
                else:
                    print(f"FAILED: {result['error']}")
                    failed += 1

    # Summary
    print()
    print("=" * 70)
    print(f"Conversion Summary:")
    print(f"  Total files: {len(pdf_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

    if successful > 0:
        total_pdf_size = sum(r['pdf_size'] for r in results if r['success'])
        total_svg_size = sum(r['svg_size'] for r in results if r['success'])
        print(f"  Total PDF size: {total_pdf_size:,} bytes")
        print(f"  Total SVG size: {total_svg_size:,} bytes")

    if failed > 0:
        print()
        print("Failed files:")
        for result in results:
            if not result['success']:
                print(f"  {result['input']}: {result['error']}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
