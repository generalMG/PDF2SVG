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
                 angle_tolerance: float = 2.0,
                 radius_tolerance: float = 0.02,
                 min_arc_points: int = 4,
                 enable_smoothing: bool = True,
                 smoothing_window: int = 5,
                 merge_dist_threshold_multiplier: float = 2.0,
                 merge_center_dist_threshold: float = 0.1,
                 merge_radius_diff_threshold: float = 0.1,
                 zigzag_len_epsilon: float = 1e-6,
                 zigzag_alternation_ratio: float = 0.5,
                 zigzag_min_angle: float = 0.1,
                 smoothing_lambda: float = 0.4,
                 smoothing_mu: float = -0.42,
                 smoothing_passes: int = 6,
                 force_smoothing: bool = False,
                 curvature_cross_threshold: float = 0.05,
                 min_radius: float = 5.0,
                 full_circle_dist_threshold_multiplier: float = 1.2,
                 full_circle_angle_span: float = 358.0,
                 least_squares_epsilon: float = 1e-10):
        """
        Args:
            arc_detection: Enable arc detection from polylines
            angle_tolerance: Max angle deviation for arc detection (degrees)
            radius_tolerance: Max relative radius deviation (0.02 = 2%)
            min_arc_points: Minimum points to consider as arc
            enable_smoothing: Enable zigzag smoothing preprocessing
            smoothing_window: Window size for moving average smoothing
            merge_dist_threshold_multiplier: Multiplier for merge distance threshold
            merge_center_dist_threshold: Threshold for merging centers
            merge_radius_diff_threshold: Threshold for merging radii
            zigzag_len_epsilon: Epsilon for zigzag length
            zigzag_alternation_ratio: Ratio for zigzag alternation
            zigzag_min_angle: Minimum angle for zigzag detection
            smoothing_lambda: Lambda for Taubin smoothing
            smoothing_mu: Mu for Taubin smoothing
            smoothing_passes: Number of passes for Taubin smoothing
            force_smoothing: Always smooth polylines before arc detection (ignore zigzag check)
            curvature_cross_threshold: Threshold for curvature cross product
            min_radius: Minimum radius for arc detection
            full_circle_dist_threshold_multiplier: Multiplier for full circle distance threshold
            full_circle_angle_span: Minimum angle span for full circle
            least_squares_epsilon: Epsilon for least squares fitting
        """
        self.arc_detection = arc_detection
        self.detector = ArcDetector(
            angle_tolerance=angle_tolerance,
            radius_tolerance=radius_tolerance,
            min_arc_points=min_arc_points,
            enable_smoothing=enable_smoothing,
            smoothing_window=smoothing_window,
            merge_dist_threshold_multiplier=merge_dist_threshold_multiplier,
            merge_center_dist_threshold=merge_center_dist_threshold,
            merge_radius_diff_threshold=merge_radius_diff_threshold,
            zigzag_len_epsilon=zigzag_len_epsilon,
            zigzag_alternation_ratio=zigzag_alternation_ratio,
            zigzag_min_angle=zigzag_min_angle,
            smoothing_lambda=smoothing_lambda,
            smoothing_mu=smoothing_mu,
            smoothing_passes=smoothing_passes,
            force_smoothing=force_smoothing,
            curvature_cross_threshold=curvature_cross_threshold,
            min_radius=min_radius,
            full_circle_dist_threshold_multiplier=full_circle_dist_threshold_multiplier,
            full_circle_angle_span=full_circle_angle_span,
            least_squares_epsilon=least_squares_epsilon
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
        # Note: PyMuPDF uses bottom-left origin (Y up), SVG uses top-left (Y down)
        # We handle coordinate transformation in each element rendering function
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
            'line_segments_saved': 0,
            'merged_circles': 0
        }

        # First pass: collect all detected arcs
        all_detected_arcs = []
        unprocessed_elements = []  # Store elements that aren't arcs

        for path_idx, path in enumerate(paths):
            detected_arcs, other_elements = self._collect_arcs_from_path(path, page_height)
            all_detected_arcs.extend(detected_arcs)
            unprocessed_elements.extend(other_elements)

        # Merge arcs that form complete circles
        merged_circles, remaining_arcs = self._merge_arcs_into_circles(all_detected_arcs)

        # Add merged circles to SVG
        for circle_info in merged_circles:
            self._add_circle(svg, circle_info['center'], circle_info['radius'],
                           page_height, circle_info['stroke'], circle_info['fill'],
                           circle_info['stroke_width'])
            stats['circles'] += 1
            stats['merged_circles'] += 1
            stats['line_segments_saved'] += circle_info['segments_saved']

        # Add remaining arcs to SVG
        for arc_info in remaining_arcs:
            if arc_info['is_circle']:
                self._add_circle(svg, arc_info['center'], arc_info['radius'],
                               page_height, arc_info['stroke'], arc_info['fill'],
                               arc_info['stroke_width'])
                stats['circles'] += 1
            else:
                self._add_arc(svg, arc_info['arc'], page_height,
                            arc_info['stroke'], arc_info['fill'],
                            arc_info['stroke_width'])
                stats['arcs'] += 1
            stats['line_segments_saved'] += arc_info['segments_saved']

        # Add other elements (lines, polylines) to SVG
        for elem in unprocessed_elements:
            if elem['type'] == 'line':
                self._add_line(svg, elem['p1'], elem['p2'], page_height,
                             elem['stroke'], elem['fill'], elem['stroke_width'])
                stats['lines'] += 1
            elif elem['type'] == 'polyline':
                self._add_polyline(svg, elem['points'], page_height,
                                  elem['stroke'], elem['fill'], elem['stroke_width'])
                stats['polylines'] += 1

        # Add statistics as comment
        stats_comment = ET.Comment(
            f" Conversion Statistics: "
            f"{stats['circles']} circles ({stats['merged_circles']} merged), "
            f"{stats['arcs']} arcs, "
            f"{stats['lines']} lines, "
            f"{stats['polylines']} polylines, "
            f"~{stats['line_segments_saved']} segments optimized "
        )
        svg.insert(0, stats_comment)

        return svg

    def _collect_arcs_from_path(self, path: Dict[str, Any], page_height: float):
        """
        Collect detected arcs and other elements from a path
        Returns: (detected_arcs, other_elements)
        """
        detected_arcs = []
        other_elements = []

        items = path.get('items', [])
        if not items:
            return detected_arcs, other_elements

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

            # Optionally smooth upfront so both arc detection and fallback polylines use it
            if self.detector:
                polyline = self.detector._maybe_smooth(polyline)

            # Try arc detection if enabled
            if self.arc_detection and len(polyline) >= self.detector.min_arc_points:
                arcs = []

                # HYBRID APPROACH: Try global circle detection first (fast preprocessing)
                # This catches high-resolution circles that AASR might miss
                circle = self.detector.detect_circle_global(polyline)
                if circle:
                    arcs = [circle]
                else:
                    # Fall back to AASR for complex/partial arcs
                    arcs = self.detector.detect_arcs(polyline)

                if arcs:
                    points_covered = 0
                    for arc in arcs:
                        arc_type = self.detector.classify_arc(arc)
                        detected_arcs.append({
                            'arc': arc,
                            'is_circle': arc_type == 'full_circle',
                            'center': arc.center.to_tuple(),
                            'radius': arc.radius,
                            'stroke': stroke,
                            'fill': fill,
                            'stroke_width': stroke_width,
                            'segments_saved': len(arc.points) - 1
                        })
                        points_covered += len(arc.points)

                    # If we covered most of the polyline, don't add as polyline
                    coverage = points_covered / len(polyline)
                    if coverage > 0.5:
                        continue

            # Fallback: add as polyline or lines
            if len(polyline) == 2:
                other_elements.append({
                    'type': 'line',
                    'p1': polyline[0],
                    'p2': polyline[1],
                    'stroke': stroke,
                    'fill': fill,
                    'stroke_width': stroke_width
                })
            else:
                other_elements.append({
                    'type': 'polyline',
                    'points': polyline,
                    'stroke': stroke,
                    'fill': fill,
                    'stroke_width': stroke_width
                })

        return detected_arcs, other_elements

    def _merge_arcs_into_circles(self, arcs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Merge multiple arcs with same center and radius into complete circles
        Returns: (merged_circles, remaining_arcs)
        """
        if not arcs:
            return [], []

        # Group arcs by center and radius
        arc_groups = {}
        for arc_info in arcs:
            cx, cy = arc_info['center']
            r = arc_info['radius']

            # Find matching group (within tolerance)
            found_group = False
            for key in list(arc_groups.keys()):
                key_cx, key_cy, key_r = key
                # Check if center and radius are similar (within 3% tolerance)
                if (abs(cx - key_cx) < r * 0.03 and
                    abs(cy - key_cy) < r * 0.03 and
                    abs(r - key_r) < r * 0.03):
                    arc_groups[key].append(arc_info)
                    found_group = True
                    break

            if not found_group:
                arc_groups[(cx, cy, r)] = [arc_info]

        merged_circles = []
        remaining_arcs = []

        # Check each group to see if arcs combine into a circle
        for (cx, cy, r), group_arcs in arc_groups.items():
            # Calculate total angle coverage
            total_angle_coverage = 0
            total_segments_saved = 0

            for arc_info in group_arcs:
                arc = arc_info['arc']
                angle_span = (arc.end_angle - arc.start_angle) % 360
                total_angle_coverage += angle_span
                total_segments_saved += arc_info['segments_saved']

            # If multiple arcs cover close to 360 degrees, merge into circle
            if len(group_arcs) >= 2 and total_angle_coverage >= 340:
                # Take stroke properties from first arc
                first_arc = group_arcs[0]
                merged_circles.append({
                    'center': (cx, cy),
                    'radius': r,
                    'stroke': first_arc['stroke'],
                    'fill': first_arc['fill'],
                    'stroke_width': first_arc['stroke_width'],
                    'segments_saved': total_segments_saved
                })
            else:
                # Keep as individual arcs
                remaining_arcs.extend(group_arcs)

        return merged_circles, remaining_arcs

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

    def _validate_curve_data(self, item: Any) -> bool:
        """
        Validate that curve/arc operation contains all required point data

        Args:
            item: PDF path item (operation, point1, point2, ...)

        Returns:
            True if all required points are present and valid, False otherwise
        """
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return False

        operation = item[0]

        try:
            if operation == 'c':  # Cubic Bezier: needs 4 points (start, ctrl1, ctrl2, end)
                if len(item) < 5:
                    return False
                # Check all points are valid
                for i in range(1, 5):
                    point = item[i]
                    if point is None:
                        return False
                    # Try to parse the point
                    parsed = self._parse_point(point)
                    if parsed is None or len(parsed) != 2:
                        return False
                return True

            elif operation == 'qu':  # Quadratic Bezier: needs 3 points (start, ctrl, end)
                if len(item) < 4:
                    return False
                # Check all points are valid
                for i in range(1, 4):
                    point = item[i]
                    if point is None:
                        return False
                    parsed = self._parse_point(point)
                    if parsed is None or len(parsed) != 2:
                        return False
                return True

            elif operation == 'v' or operation == 'y':  # Bezier variants
                # These are special Bezier forms, need validation
                if len(item) < 3:
                    return False
                for i in range(1, len(item)):
                    point = item[i]
                    if point is None:
                        return False
                    parsed = self._parse_point(point)
                    if parsed is None or len(parsed) != 2:
                        return False
                return True

        except (ValueError, TypeError, AttributeError):
            return False

        return False

    def _extract_polylines(self, items: List[Any]) -> List[List[Tuple[float, float]]]:
        """
        Extract connected sequences of line segments (polylines) from path items
        PyMuPDF items format: ('l', Point(x1, y1), Point(x2, y2)) for lines

        Also validates curve operations and falls back to line extraction if incomplete
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
                # Check if it's a curve operation with complete data
                if operation in ('c', 'qu', 'v', 'y'):
                    if self._validate_curve_data(item):
                        # TODO: In future, handle these curve operations properly
                        # For now, log that we found valid curve data but skip it
                        pass
                    # else: incomplete curve data, will be skipped

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
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]

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
        points_str = ' '.join(f"{p[0]:.3f},{p[1]:.3f}" for p in points)

        ET.SubElement(svg, 'polyline', {
            'points': points_str,
            'stroke': stroke,
            'stroke-width': f"{stroke_width}",
            'fill': fill
        })

    def _add_circle(self, svg: ET.Element, center: Tuple[float, float], radius: float,
                   page_height: float, stroke: str, fill: str, stroke_width: float):
        """Add a circle element to SVG"""
        cx, cy = center[0], center[1]

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
        cx, cy = arc.center.x, arc.center.y
        r = arc.radius

        # Calculate start and end points
        start_angle_rad = math.radians(arc.start_angle)
        end_angle_rad = math.radians(arc.end_angle)

        start_x = cx + r * math.cos(start_angle_rad)
        start_y = cy + r * math.sin(start_angle_rad)
        end_x = cx + r * math.cos(end_angle_rad)
        end_y = cy + r * math.sin(end_angle_rad)

        # Calculate angle span
        angle_span = (arc.end_angle - arc.start_angle) % 360

        # Determine large-arc-flag
        large_arc = 1 if angle_span > 180 else 0

        # Determine sweep-flag
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
    parser.add_argument('--angle_tolerance', type=float, default=2.0,
                       help='Angle tolerance for arc detection (degrees, default: 2.0)')
    parser.add_argument('--radius_tolerance', type=float, default=0.02,
                       help='Radius tolerance for arc detection (fraction, default: 0.02)')
    parser.add_argument('--min_arc_points', type=int, default=4,
                       help='Minimum points to consider as arc (default: 4)')
    parser.add_argument('--no-smoothing', action='store_true',
                       help='Disable zigzag smoothing preprocessing')
    parser.add_argument('--smoothing_window', type=int, default=5,
                       help='Window size for smoothing (must be odd, default: 5)')
    parser.add_argument('--force-smoothing', action='store_true',
                       help='Force smoothing even if zigzag detection does not trigger')
    
    # New arguments
    parser.add_argument('--merge_dist_threshold_multiplier', type=float, default=2.0, help='Multiplier for merge distance threshold')
    parser.add_argument('--merge_center_dist_threshold', type=float, default=0.1, help='Threshold for merging centers')
    parser.add_argument('--merge_radius_diff_threshold', type=float, default=0.1, help='Threshold for merging radii')
    parser.add_argument('--zigzag_len_epsilon', type=float, default=1e-6, help='Epsilon for zigzag length')
    parser.add_argument('--zigzag_alternation_ratio', type=float, default=0.5, help='Ratio for zigzag alternation')
    parser.add_argument('--zigzag_min_angle', type=float, default=0.1, help='Minimum angle for zigzag detection')
    parser.add_argument('--smoothing_lambda', type=float, default=0.4, help='Lambda for Taubin smoothing')
    parser.add_argument('--smoothing_mu', type=float, default=-0.42, help='Mu for Taubin smoothing')
    parser.add_argument('--smoothing_passes', type=int, default=6, help='Number of passes for Taubin smoothing')
    parser.add_argument('--curvature_cross_threshold', type=float, default=0.05, help='Threshold for curvature cross product')
    parser.add_argument('--min_radius', type=float, default=5.0, help='Minimum radius for arc detection')
    parser.add_argument('--full_circle_dist_threshold_multiplier', type=float, default=1.2, help='Multiplier for full circle distance threshold')
    parser.add_argument('--full_circle_angle_span', type=float, default=358.0, help='Minimum angle span for full circle')
    parser.add_argument('--least_squares_epsilon', type=float, default=1e-10, help='Epsilon for least squares fitting')

    args = parser.parse_args()

    # Create converter
    converter = PDFtoSVGConverter(
        arc_detection=not args.no_arc_detection,
        angle_tolerance=args.angle_tolerance,
        radius_tolerance=args.radius_tolerance,
        min_arc_points=args.min_arc_points,
        enable_smoothing=not args.no_smoothing,
        smoothing_window=args.smoothing_window,
        force_smoothing=args.force_smoothing,
        merge_dist_threshold_multiplier=args.merge_dist_threshold_multiplier,
        merge_center_dist_threshold=args.merge_center_dist_threshold,
        merge_radius_diff_threshold=args.merge_radius_diff_threshold,
        zigzag_len_epsilon=args.zigzag_len_epsilon,
        zigzag_alternation_ratio=args.zigzag_alternation_ratio,
        zigzag_min_angle=args.zigzag_min_angle,
        smoothing_lambda=args.smoothing_lambda,
        smoothing_mu=args.smoothing_mu,
        smoothing_passes=args.smoothing_passes,
        curvature_cross_threshold=args.curvature_cross_threshold,
        min_radius=args.min_radius,
        full_circle_dist_threshold_multiplier=args.full_circle_dist_threshold_multiplier,
        full_circle_angle_span=args.full_circle_angle_span,
        least_squares_epsilon=args.least_squares_epsilon
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
