#!/usr/bin/env python3
"""
PDF to SVG Converter with Arc Reconstruction
Converts PDF technical drawings to SVG, detecting and preserving arcs/circles
"""

import fitz  # PyMuPDF
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import math
from arc_detector import ArcDetector, parse_point_string


class PDFtoSVGConverter:
    def __init__(self, arc_detection: bool = True,
                 angle_tolerance: float = 8.0,
                 radius_tolerance: float = 0.03,
                 min_arc_points: int = 4):
        """
        Args:
            arc_detection: Enable arc detection from polylines
            angle_tolerance: Max angle deviation for arc detection (degrees)
            radius_tolerance: Max relative radius deviation (0.03 = 3%)
            min_arc_points: Minimum points to consider as arc
        """
        self.arc_detection = arc_detection
        self.detector = ArcDetector(
            angle_tolerance=angle_tolerance,
            radius_tolerance=radius_tolerance,
            min_arc_points=min_arc_points
        ) if arc_detection else None

    def convert(self, pdf_path: str, output_svg: str = None, page_num: int = 0) -> str:
        """
        Convert PDF page to SVG

        Args:
            pdf_path: Path to input PDF
            output_svg: Path to output SVG (optional, defaults to output/<filename>.svg)
            page_num: Page number to convert (0-indexed)

        Returns:
            Path to generated SVG file
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if output_svg is None:
            # Default to output directory
            output_dir = Path('output')
            output_dir.mkdir(parents=True, exist_ok=True)
            output_svg = output_dir / pdf_path.with_suffix('.svg').name
        else:
            output_svg = Path(output_svg)
            # Create parent directories if needed
            output_svg.parent.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(pdf_path))

        if page_num >= len(doc):
            raise ValueError(f"Page {page_num} not found (PDF has {len(doc)} pages)")

        page = doc[page_num]
        svg_root = self._convert_page(page, pdf_path.stem)

        doc.close()

        # Write SVG
        self._write_svg(svg_root, output_svg)

        return str(output_svg)

    def _convert_page(self, page: fitz.Page, title: str = "") -> ET.Element:
        """Convert a PDF page to SVG ElementTree"""

        page_width = page.rect.width
        page_height = page.rect.height

        # Create SVG root
        svg = ET.Element('svg', {
            'xmlns': 'http://www.w3.org/2000/svg',
            'width': f"{page_width}",
            'height': f"{page_height}",
            'viewBox': f"0 0 {page_width} {page_height}",
            'version': '1.1'
        })

        # Add title
        if title:
            title_elem = ET.SubElement(svg, 'title')
            title_elem.text = title

        # Add description
        desc = ET.SubElement(svg, 'desc')
        desc.text = f"Converted from PDF with arc reconstruction"

        # Extract drawing paths
        paths = page.get_drawings()

        # Statistics
        stats = {
            'total_paths': len(paths),
            'lines': 0,
            'arcs': 0,
            'circles': 0,
            'polylines': 0,
            'line_segments_saved': 0
        }

        # Process each path
        for path_idx, path in enumerate(paths):
            layer_name = f"path_{path_idx}"
            self._process_path(svg, path, page_height, layer_name, stats)

        # Add statistics as comment
        stats_comment = ET.Comment(
            f" Conversion Statistics: "
            f"{stats['circles']} circles, "
            f"{stats['arcs']} arcs, "
            f"{stats['lines']} lines, "
            f"{stats['polylines']} polylines, "
            f"~{stats['line_segments_saved']} segments optimized "
        )
        svg.insert(0, stats_comment)

        return svg

    def _process_path(self, svg: ET.Element, path: Dict[str, Any],
                     page_height: float, layer: str, stats: Dict[str, int]):
        """Process a single PDF path and add elements to SVG"""

        items = path.get('items', [])
        if not items:
            return

        # Extract stroke properties
        stroke_color = path.get('color')
        fill_color = path.get('fill')
        stroke_width = path.get('width', 1.0)

        stroke = self._rgb_to_hex(stroke_color) if stroke_color else 'none'
        fill = self._rgb_to_hex(fill_color) if fill_color else 'none'

        # Group consecutive line segments into polylines
        polylines = self._extract_polylines(items)

        # Process each polyline
        for polyline in polylines:
            if len(polyline) < 2:
                continue

            # Try arc detection if enabled
            if self.arc_detection and len(polyline) >= self.detector.min_arc_points:
                elements_added = self._try_add_arcs(
                    svg, polyline, page_height, stroke, fill, stroke_width, stats
                )
                if elements_added:
                    continue

            # Fallback: add as polyline or lines
            if len(polyline) == 2:
                # Single line
                self._add_line(svg, polyline[0], polyline[1], page_height,
                             stroke, fill, stroke_width)
                stats['lines'] += 1
            else:
                # Multiple connected lines
                self._add_polyline(svg, polyline, page_height, stroke, fill, stroke_width)
                stats['polylines'] += 1

    def _extract_polylines(self, items: List[Any]) -> List[List[Tuple[float, float]]]:
        """
        Extract connected sequences of line segments (polylines) from path items
        PyMuPDF items format: ('l', Point(x1, y1), Point(x2, y2)) for lines
        """
        polylines = []
        current_polyline = []
        current_pos = None

        for item in items:
            # Item is a tuple: (operation, point1, point2, ...)
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue

            operation = item[0]

            if operation != 'l':
                # Non-line item, end current polyline
                if len(current_polyline) >= 2:
                    polylines.append(current_polyline)
                current_polyline = []
                current_pos = None
                continue

            # Line format: ('l', from_point, to_point)
            from_pt = self._parse_point(item[1])
            to_pt = self._parse_point(item[2])

            if current_pos is None:
                # Start new polyline
                current_polyline = [from_pt, to_pt]
                current_pos = to_pt
            else:
                # Check if this line connects to the current polyline
                dist_to_start = self._point_distance(from_pt, current_polyline[-1])
                dist_to_end = self._point_distance(to_pt, current_polyline[-1])

                if dist_to_start < 0.1:  # Connected to end
                    current_polyline.append(to_pt)
                    current_pos = to_pt
                elif dist_to_end < 0.1:  # Reverse connection
                    current_polyline.append(from_pt)
                    current_pos = from_pt
                else:
                    # Not connected, start new polyline
                    if len(current_polyline) >= 2:
                        polylines.append(current_polyline)
                    current_polyline = [from_pt, to_pt]
                    current_pos = to_pt

        # Add last polyline
        if len(current_polyline) >= 2:
            polylines.append(current_polyline)

        return polylines

    def _try_add_arcs(self, svg: ET.Element, polyline: List[Tuple[float, float]],
                     page_height: float, stroke: str, fill: str, stroke_width: float,
                     stats: Dict[str, int]) -> bool:
        """
        Try to detect and add arcs from polyline
        Returns True if arcs were added, False otherwise
        """
        arcs = self.detector.detect_arcs(polyline)

        if not arcs:
            return False

        points_covered = 0

        for arc in arcs:
            arc_type = self.detector.classify_arc(arc)

            if arc_type == 'full_circle':
                # Add as circle element
                self._add_circle(svg, arc.center.to_tuple(), arc.radius,
                               page_height, stroke, fill, stroke_width)
                stats['circles'] += 1
                points_covered += len(arc.points)
                stats['line_segments_saved'] += len(arc.points) - 1

            else:
                # Add as arc (using SVG path)
                self._add_arc(svg, arc, page_height, stroke, fill, stroke_width)
                stats['arcs'] += 1
                points_covered += len(arc.points)
                stats['line_segments_saved'] += len(arc.points) - 1

        # If we didn't cover most of the polyline, return False
        coverage = points_covered / len(polyline)
        return coverage > 0.5

    def _add_line(self, svg: ET.Element, p1: Tuple[float, float], p2: Tuple[float, float],
                  page_height: float, stroke: str, fill: str, stroke_width: float):
        """Add a line element to SVG"""
        x1, y1 = p1[0], page_height - p1[1]
        x2, y2 = p2[0], page_height - p2[1]

        ET.SubElement(svg, 'line', {
            'x1': f"{x1:.3f}",
            'y1': f"{y1:.3f}",
            'x2': f"{x2:.3f}",
            'y2': f"{y2:.3f}",
            'stroke': stroke,
            'stroke-width': f"{stroke_width}",
            'fill': 'none'
        })

    def _add_polyline(self, svg: ET.Element, points: List[Tuple[float, float]],
                     page_height: float, stroke: str, fill: str, stroke_width: float):
        """Add a polyline element to SVG"""
        points_str = ' '.join(f"{p[0]:.3f},{page_height - p[1]:.3f}" for p in points)

        ET.SubElement(svg, 'polyline', {
            'points': points_str,
            'stroke': stroke,
            'stroke-width': f"{stroke_width}",
            'fill': fill
        })

    def _add_circle(self, svg: ET.Element, center: Tuple[float, float], radius: float,
                   page_height: float, stroke: str, fill: str, stroke_width: float):
        """Add a circle element to SVG"""
        cx, cy = center[0], page_height - center[1]

        ET.SubElement(svg, 'circle', {
            'cx': f"{cx:.3f}",
            'cy': f"{cy:.3f}",
            'r': f"{radius:.3f}",
            'stroke': stroke,
            'stroke-width': f"{stroke_width}",
            'fill': fill
        })

    def _add_arc(self, svg: ET.Element, arc: Any, page_height: float,
                stroke: str, fill: str, stroke_width: float):
        """Add an arc using SVG path element"""
        # Convert center and angles to SVG arc parameters
        cx, cy = arc.center.x, page_height - arc.center.y
        r = arc.radius

        # Calculate start and end points
        start_angle_rad = math.radians(arc.start_angle)
        end_angle_rad = math.radians(arc.end_angle)

        # Y is flipped, so we need to negate angles
        start_x = cx + r * math.cos(start_angle_rad)
        start_y = cy - r * math.sin(start_angle_rad)
        end_x = cx + r * math.cos(end_angle_rad)
        end_y = cy - r * math.sin(end_angle_rad)

        # Calculate angle span
        angle_span = (arc.end_angle - arc.start_angle) % 360

        # Determine large-arc-flag
        large_arc = 1 if angle_span > 180 else 0

        # Determine sweep-flag (clockwise in SVG coordinates)
        # Since Y is flipped, we need to invert the sweep
        sweep = 1 if angle_span > 0 else 0

        # Create path data
        path_data = (
            f"M {start_x:.3f},{start_y:.3f} "
            f"A {r:.3f},{r:.3f} 0 {large_arc},{sweep} {end_x:.3f},{end_y:.3f}"
        )

        ET.SubElement(svg, 'path', {
            'd': path_data,
            'stroke': stroke,
            'stroke-width': f"{stroke_width}",
            'fill': fill
        })

    def _parse_point(self, point: Any) -> Tuple[float, float]:
        """Parse point from various formats"""
        if isinstance(point, (list, tuple)):
            return (float(point[0]), float(point[1]))
        elif isinstance(point, str):
            return parse_point_string(point)
        elif hasattr(point, 'x') and hasattr(point, 'y'):
            return (float(point.x), float(point.y))
        else:
            raise ValueError(f"Unknown point format: {point}")

    def _point_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _rgb_to_hex(self, rgb: List[float]) -> str:
        """Convert RGB values [0-1] to hex color"""
        if rgb is None:
            return 'none'
        r, g, b = [int(c * 255) for c in rgb[:3]]
        return f"#{r:02x}{g:02x}{b:02x}"

    def _write_svg(self, svg_root: ET.Element, output_path: str):
        """Write SVG to file with pretty formatting"""
        # Convert to string
        xml_str = ET.tostring(svg_root, encoding='unicode')

        # Pretty print
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent='  ')

        # Remove extra blank lines
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        pretty_xml = '\n'.join(lines)

        # Write to file
        with open(output_path, 'w') as f:
            f.write(pretty_xml)


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert PDF technical drawings to SVG with arc reconstruction'
    )
    parser.add_argument('pdf_file', help='Input PDF file')
    parser.add_argument('-o', '--output', help='Output SVG file (default: output/<filename>.svg)')
    parser.add_argument('-p', '--page', type=int, default=0, help='Page number (0-indexed, default: 0)')
    parser.add_argument('--no-arc-detection', action='store_true',
                       help='Disable arc detection (output polylines only)')
    parser.add_argument('--angle-tolerance', type=float, default=8.0,
                       help='Angle tolerance for arc detection (degrees, default: 8.0)')
    parser.add_argument('--radius-tolerance', type=float, default=0.03,
                       help='Radius tolerance for arc detection (fraction, default: 0.03)')
    parser.add_argument('--min-arc-points', type=int, default=4,
                       help='Minimum points to consider as arc (default: 4)')

    args = parser.parse_args()

    # Create converter
    converter = PDFtoSVGConverter(
        arc_detection=not args.no_arc_detection,
        angle_tolerance=args.angle_tolerance,
        radius_tolerance=args.radius_tolerance,
        min_arc_points=args.min_arc_points
    )

    try:
        output_svg = converter.convert(args.pdf_file, args.output, args.page)
        print(f"Successfully converted: {args.pdf_file}")
        print(f"Output SVG: {output_svg}")

        # Show file size comparison
        pdf_size = Path(args.pdf_file).stat().st_size
        svg_size = Path(output_svg).stat().st_size
        print(f"\nFile sizes:")
        print(f"  PDF: {pdf_size:,} bytes")
        print(f"  SVG: {svg_size:,} bytes")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
