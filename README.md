# PDF2SVG - Technical Drawing Converter

A high-quality PDF to SVG converter specifically designed for technical drawings and CAD documents. This tool intelligently detects and reconstructs circular arcs from polyline approximations, preserving geometric precision for downstream CAD workflows.

## Problem Statement

Technical drawings exported as PDF often represent circles and arcs as many small line segments (polylines). This approximation causes issues when converting to CAD formats like DXF:

- Loss of geometric precision
- Unnecessarily large file sizes
- Difficulty in CAD software manipulation
- No parametric arc information (center, radius, angles)

## Solution

PDF2SVG analyzes line segment sequences and reconstructs them as proper geometric primitives:

- **Arc Detection**: Identifies sequences of connected line segments forming circular arcs
- **Circle Reconstruction**: Calculates center, radius, and angular parameters
- **Geometric Preservation**: Outputs SVG with true arc and circle elements
- **High Compatibility**: SVG output ready for DXF conversion using existing tools

## Features

- Intelligent arc detection from polyline approximations
- Configurable detection sensitivity (angle tolerance, radius tolerance)
- Preserves non-arc geometry (straight lines, polylines)
- Batch processing with parallel execution
- Automatic output directory management
- Detailed conversion statistics
- Compatible with downstream SVG-to-DXF converters
- No external dependencies beyond PyMuPDF

## Project Structure

```
PDF2SVG/
├── pdf_to_svg.py          # Main converter (single file)
├── batch_convert.py       # Batch processor (parallel)
├── arc_detector.py        # Arc detection algorithm
├── analyze_pdf.py         # PDF analysis tool
├── requirements.txt       # Dependencies
├── README.md              # Full documentation
├── QUICKSTART.md          # Quick start guide
└── output/                # Default output directory (auto-created)
```

## Installation

### Requirements

- Python 3.7+
- PyMuPDF (fitz)

### Setup

```bash
pip install PyMuPDF
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

## Usage

### Single File Conversion

Convert a single PDF to SVG:

```bash
python pdf_to_svg.py input.pdf
```

This creates `output/input.svg` in the output directory (created automatically).

Specify custom output location:

```bash
python pdf_to_svg.py input.pdf -o custom/path/output.svg
```

### Options

```
positional arguments:
  pdf_file              Input PDF file path

optional arguments:
  -h, --help            Show help message
  -o, --output OUTPUT   Output SVG file path (default: output/<filename>.svg)
  -p, --page PAGE       Page number to convert, 0-indexed (default: 0)
  --no-arc-detection    Disable arc detection, output polylines only
  --angle-tolerance DEG Angle tolerance for arc detection in degrees (default: 8.0)
  --radius-tolerance    Radius tolerance as fraction (default: 0.03 = 3%)
  --min-arc-points N    Minimum points to consider as arc (default: 4)
```

### Examples

Convert specific page with custom tolerances:

```bash
python pdf_to_svg.py drawing.pdf -p 1 --angle-tolerance 10 --radius-tolerance 0.05
```

Disable arc detection (output raw polylines):

```bash
python pdf_to_svg.py drawing.pdf --no-arc-detection
```

### Batch Conversion

Convert multiple PDFs in parallel:

```bash
python batch_convert.py input_dir/ -j 8
```

This creates all SVG files in `output/` directory (default).

Specify custom output directory:

```bash
python batch_convert.py file1.pdf file2.pdf dir1/ -o custom_output/
```

Batch options:

```
positional arguments:
  inputs                Input PDF files or directories (searches recursively)

optional arguments:
  -o, --output-dir DIR  Output directory for SVG files (default: output/)
  -j, --jobs N          Number of parallel jobs (default: 4)
  --angle-tolerance     Arc detection angle tolerance (default: 8.0)
  --radius-tolerance    Arc detection radius tolerance (default: 0.03)
  --min-arc-points      Minimum points for arc (default: 4)
  --no-arc-detection    Disable arc detection
```

## Arc Detection Algorithm

The arc detector uses geometric analysis to identify circular arcs:

1. **Polyline Extraction**: Groups connected line segments
2. **Collinearity Check**: Filters out straight lines
3. **Circle Fitting**: Calculates center from three points using perpendicular bisectors
4. **Arc Extension**: Extends arc to include all co-circular points
5. **Validation**: Checks radius consistency and angular spacing
6. **Classification**: Identifies full circles, semicircles, major/minor arcs

### Detection Parameters

- **Angle Tolerance**: Maximum angular deviation between consecutive segments (default: 8 degrees)
- **Radius Tolerance**: Maximum relative radius variation across arc points (default: 3%)
- **Minimum Arc Points**: Minimum segments to consider as arc candidate (default: 4 points)

### Tuning Recommendations

For **high-precision drawings** (many segments per arc):
```bash
--angle-tolerance 5.0 --radius-tolerance 0.02 --min-arc-points 6
```

For **low-resolution drawings** (few segments per arc):
```bash
--angle-tolerance 12.0 --radius-tolerance 0.05 --min-arc-points 3
```

## Output Format

Generated SVG files contain:

- **Circle elements**: `<circle cx="..." cy="..." r="..."/>`
- **Arc paths**: `<path d="M ... A ... "/>` with proper arc parameters
- **Polylines**: `<polyline points="..."/>` for connected segments
- **Lines**: `<line x1="..." y1="..." x2="..." y2="..."/>` for single segments
- **Statistics comment**: Conversion summary in XML comment

### Conversion Statistics

Each SVG includes a statistics comment showing optimization results:

```xml
<!-- Conversion Statistics: 0 circles, 1131 arcs, 64 lines, 563 polylines, ~4793 segments optimized -->
```

## Integration with SVG-to-DXF Pipeline

This tool is designed to work with the VectorImgAnalysis SVG-to-DXF converter:

### Complete Workflow

1. **PDF to SVG** (this tool):
```bash
python pdf_to_svg.py input.pdf
```

This creates `output/input.svg`

2. **SVG to DXF** (VectorImgAnalysis):
```bash
cd /mnt/d/mg_ai_research/workspace/whatnot/VectorImgAnalysis
python svg_to_dxf.py /mnt/d/mg_ai_research/workspace/whatnot/PDF2SVG/output/input.svg -o final.dxf
```

### Expected Benefits

- **Reduced segment count**: Thousands of line segments replaced by arc primitives
- **Parametric arcs**: DXF files contain true arc entities with center/radius
- **Smaller file sizes**: More compact representation
- **CAD compatibility**: Arcs recognized as editable curves in CAD software

## Architecture

### Core Components

**arc_detector.py**: Arc detection and geometric analysis
- `ArcDetector`: Main detection engine
- Circle center calculation using perpendicular bisectors
- Collinearity testing and radius validation
- Arc classification (full circle, major/minor arc)

**pdf_to_svg.py**: PDF parsing and SVG generation
- `PDFtoSVGConverter`: Main converter class
- PyMuPDF integration for PDF vector extraction
- Polyline grouping and arc detection
- SVG element generation with proper attributes

**batch_convert.py**: Parallel batch processing
- Multi-file processing with process pools
- Progress reporting and error handling
- Recursive directory scanning

**analyze_pdf.py**: PDF structure analysis tool
- Extract raw drawing commands
- Analyze path composition
- Generate detailed JSON reports

## Testing

### Test Arc Detection Algorithm

```bash
python arc_detector.py
```

This runs built-in tests with synthetic circle and arc data.

### Analyze PDF Structure

```bash
python analyze_pdf.py
```

Analyzes the sample PDFs (hardcoded paths in script) and generates in the current directory:
- `*_analysis.json`: Detailed path structure
- `*_raw_commands.txt`: Raw PDF drawing commands

To analyze your own PDF, modify the `pdf_files` list in `analyze_pdf.py`.

### Test Files

Example conversions (outputs to `output/` directory):

```bash
python pdf_to_svg.py "60355K178_Ball Bearing.pdf"
python pdf_to_svg.py "61355K31_Combination Clutch Brake.pdf"
```

Expected results:
- Ball Bearing: ~1131 arcs detected, ~4793 segments optimized
- Clutch Brake: ~846 arcs detected, ~3393 segments optimized

Actual results depend on arc detection parameters and PDF structure.

## Performance

### Conversion Speed

Typical processing times (single-threaded):
- Simple drawing (500 segments): < 1 second
- Complex drawing (15,000 segments): 2-5 seconds
- Very complex (50,000+ segments): 10-20 seconds

### Parallel Processing

Batch mode with 8 workers:
- 10 files: ~5-10 seconds total
- 100 files: ~30-60 seconds total

### Memory Usage

- Typical file: 10-50 MB RAM
- Large technical drawing: 100-200 MB RAM

## Troubleshooting

### No Arcs Detected

**Symptoms**: All geometry output as polylines

**Solutions**:
- Increase `--angle-tolerance` to 10-15 degrees
- Increase `--radius-tolerance` to 0.05-0.10
- Decrease `--min-arc-points` to 3

### Too Many False Arcs

**Symptoms**: Straight lines detected as arcs

**Solutions**:
- Decrease `--angle-tolerance` to 5-6 degrees
- Decrease `--radius-tolerance` to 0.01-0.02
- Increase `--min-arc-points` to 5-6

### Conversion Errors

- **Division by zero**: Fixed in current version with degenerate arc handling
- **Import errors**: Ensure PyMuPDF is installed: `pip install PyMuPDF`
- **File not found**: Use absolute paths or quotes for paths with spaces
- **Output permission denied**: Ensure write permissions for `output/` directory

## Limitations

- **Elliptical arcs**: Currently only detects circular arcs (equal X/Y radius)
- **Bezier curves**: Not detected, output as polylines
- **Text elements**: Not extracted (focuses on vector graphics)
- **Raster images**: Not included in SVG output
- **Multi-page**: Processes one page at a time (use -p flag for page selection)

## Future Enhancements

Planned features:
- Ellipse detection for non-circular arcs
- Spline/Bezier curve reconstruction
- Multi-page batch processing
- Direct PDF-to-DXF conversion (bypass SVG intermediate)
- GUI for parameter tuning and visual preview
- Support for additional vector formats (EPS, AI)

## License

This project is provided as-is for technical drawing conversion workflows.

## Contributing

Contributions welcome. Focus areas:
- Ellipse detection algorithms
- Bezier curve reconstruction
- Performance optimizations
- Additional output formats

## Project Files

### Main Scripts
- **pdf_to_svg.py** (16KB): Single file converter with CLI
- **batch_convert.py** (6KB): Parallel batch processor
- **arc_detector.py** (15KB): Geometric arc detection engine
- **analyze_pdf.py** (8KB): PDF structure analysis utility

### Documentation
- **README.md** (11KB): Complete documentation
- **QUICKSTART.md** (2KB): Quick start guide
- **requirements.txt**: PyMuPDF dependency

### Output
- **output/** directory: Auto-created for SVG files
- **\*_analysis.json**: Optional PDF analysis output
- **\*_raw_commands.txt**: Optional raw PDF commands

## Related Projects

**VectorImgAnalysis**: SVG to DXF converter
- Location: `/mnt/d/mg_ai_research/workspace/whatnot/VectorImgAnalysis`
- Handles SVG path parsing and DXF entity generation
- Supports arcs, circles, ellipses, splines

## Citation

If using this tool in research or technical work:

```
PDF2SVG - Technical Drawing Converter with Arc Reconstruction
Converts PDF vector graphics to SVG, preserving geometric primitives
```

## Support

For issues, feature requests, or questions:
- Check troubleshooting section above
- Review conversion statistics in output SVG
- Test with analyze_pdf.py to inspect PDF structure
- Adjust detection parameters based on drawing characteristics

## Acknowledgments

Built with:
- PyMuPDF (fitz) for PDF parsing
- Python xml.etree for SVG generation
- Python math library for geometric calculations

Designed for integration with VectorImgAnalysis SVG-to-DXF pipeline.
