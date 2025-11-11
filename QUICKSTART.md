# Quick Start Guide

## Basic Usage

### Convert a Single PDF

```bash
python pdf_to_svg.py "path/to/your/drawing.pdf"
```

Output: `output/drawing.svg`

### Convert Multiple PDFs

```bash
python batch_convert.py "path/to/pdf/directory/" -j 8
```

All SVG files will be in the `output/` directory.

### Convert Specific Files

```bash
python batch_convert.py file1.pdf file2.pdf file3.pdf
```

## Common Scenarios

### High-Precision Technical Drawings

For drawings with many segments per arc:

```bash
python pdf_to_svg.py drawing.pdf \
  --angle-tolerance 5.0 \
  --radius-tolerance 0.02 \
  --min-arc-points 6
```

### Low-Resolution Drawings

For drawings with few segments per arc:

```bash
python pdf_to_svg.py drawing.pdf \
  --angle-tolerance 12.0 \
  --radius-tolerance 0.05 \
  --min-arc-points 3
```

### Disable Arc Detection

Output raw polylines without arc reconstruction:

```bash
python pdf_to_svg.py drawing.pdf --no-arc-detection
```

## Complete PDF to DXF Workflow

1. Convert PDF to SVG with arc reconstruction:
```bash
python pdf_to_svg.py input.pdf
```

2. Convert SVG to DXF:
```bash
cd /path/to/VectorImgAnalysis
python svg_to_dxf.py /path/to/output/input.svg -o output.dxf
```

## Output Location

By default, all generated SVG files are saved to the `output/` directory, which is created automatically if it doesn't exist.

To specify a custom output location:

```bash
# Single file
python pdf_to_svg.py input.pdf -o custom/location/output.svg

# Batch
python batch_convert.py input_dir/ -o custom_output_dir/
```

## Checking Results

Each SVG file includes conversion statistics in an XML comment at the top:

```xml
<!-- Conversion Statistics: 0 circles, 1131 arcs, 64 lines, 563 polylines, ~4793 segments optimized -->
```

This shows how many line segments were successfully converted to arc primitives.
