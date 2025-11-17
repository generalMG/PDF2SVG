#!/usr/bin/env python3
"""
Arc Detection from Polylines
Reconstructs circular arcs and circles from line segment approximations
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class Point:
    x: float
    y: float

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def to_tuple(self):
        return (self.x, self.y)


@dataclass
class Arc:
    center: Point
    radius: float
    start_angle: float  # degrees
    end_angle: float    # degrees
    points: List[Point]
    is_full_circle: bool = False

    def arc_length(self):
        """Calculate arc length"""
        angle_diff = abs(self.end_angle - self.start_angle)
        return (angle_diff / 360.0) * 2 * math.pi * self.radius


class ArcDetector:
    def __init__(self, angle_tolerance: float = 5.0, radius_tolerance: float = 0.02,
                 min_arc_points: int = 4, collinearity_threshold: float = 0.001):
        """
        Args:
            angle_tolerance: Max deviation in degrees from expected arc angles
            radius_tolerance: Max relative deviation in radius (0.02 = 2%)
            min_arc_points: Minimum points to consider as arc
            collinearity_threshold: Threshold for detecting straight lines
        """
        self.angle_tolerance = angle_tolerance
        self.radius_tolerance = radius_tolerance
        self.min_arc_points = min_arc_points
        self.collinearity_threshold = collinearity_threshold

    def detect_arcs(self, points: List[Tuple[float, float]]) -> List[Arc]:
        """
        Detect arcs from a sequence of connected points using
        Angle-Based Curvature Segmentation and Reconstruction (AASR)

        Returns list of Arc objects
        """
        if len(points) < self.min_arc_points:
            return []

        points = [Point(p[0], p[1]) for p in points]

        # Step 1: Segment polyline by curvature regions
        curved_segments = self._segment_by_curvature(points)

        # Step 2: Fit circles/arcs to curved segments
        arcs = []
        for segment in curved_segments:
            if len(segment) >= self.min_arc_points:
                arc = self._fit_arc_to_segment(segment)
                if arc:
                    arcs.append(arc)

        return arcs

    def _segment_by_curvature(self, points: List[Point],
                              angle_tolerance_rad: float = None) -> List[List[Point]]:
        """
        Segment polyline into regions of consistent curvature using AASR algorithm

        Returns list of curved segments (filters out straight sections)
        """
        if angle_tolerance_rad is None:
            # Convert angle tolerance from degrees to radians
            angle_tolerance_rad = math.radians(self.angle_tolerance)

        if len(points) < 3:
            return []

        segments = []
        current_segment = [points[0]]
        current_curvature_sign = None

        for i in range(1, len(points) - 1):
            # Calculate vectors for consecutive segments
            v1 = Point(points[i].x - points[i-1].x, points[i].y - points[i-1].y)
            v2 = Point(points[i+1].x - points[i].x, points[i+1].y - points[i].y)

            # Normalize vectors
            len_v1 = math.sqrt(v1.x**2 + v1.y**2)
            len_v2 = math.sqrt(v2.x**2 + v2.y**2)

            if len_v1 < 1e-6 or len_v2 < 1e-6:
                # Skip degenerate segments
                continue

            v1_norm = Point(v1.x / len_v1, v1.y / len_v1)
            v2_norm = Point(v2.x / len_v2, v2.y / len_v2)

            # Calculate angle between vectors
            dot = v1_norm.x * v2_norm.x + v1_norm.y * v2_norm.y
            dot = max(-1.0, min(1.0, dot))  # Clamp to [-1, 1]
            theta = math.acos(dot)

            # Calculate curvature direction using cross product (2D)
            cross = v1_norm.x * v2_norm.y - v1_norm.y * v2_norm.x
            curvature_sign = 1 if cross > 0 else -1 if cross < 0 else 0

            # Check if this is a straight section
            # theta is the angle between consecutive vectors:
            # - theta ≈ 0 means vectors point in SAME direction (straight line)
            # - theta ≈ π means vectors point in OPPOSITE directions (sharp reversal)
            # We want to detect when theta is close to 0 (straight continuation)
            if theta < angle_tolerance_rad:
                # Straight section - end current curved segment
                if len(current_segment) >= self.min_arc_points:
                    segments.append(current_segment)
                current_segment = [points[i]]
                current_curvature_sign = None
            else:
                # Curved section
                if current_curvature_sign is None:
                    # Start new curved segment
                    current_curvature_sign = curvature_sign
                    current_segment.append(points[i])
                elif current_curvature_sign == curvature_sign:
                    # Continue current curved segment (same direction)
                    current_segment.append(points[i])
                else:
                    # Curvature direction changed - end current segment
                    if len(current_segment) >= self.min_arc_points:
                        segments.append(current_segment)
                    current_segment = [points[i]]
                    current_curvature_sign = curvature_sign

        # Add last point to current segment
        if len(points) > 0:
            current_segment.append(points[-1])

        # Add final segment if it's long enough
        if len(current_segment) >= self.min_arc_points:
            segments.append(current_segment)

        return segments

    def _fit_arc_to_segment(self, segment: List[Point]) -> Optional[Arc]:
        """
        Fit a circle/arc to an entire curved segment using global least-squares

        This is the core AASR reconstruction step
        """
        if len(segment) < 3:
            return None

        # Check if segment is collinear (straight line, not a curve)
        if len(segment) >= 3 and self._are_collinear(segment[0], segment[1], segment[2]):
            return None

        # Try to fit circle to entire segment
        # Use first, middle, and last points for initial circle estimation
        mid_idx = len(segment) // 2
        center = self._find_circle_center(segment[0], segment[mid_idx], segment[-1])

        if center is None:
            return None

        # Calculate radius from center to all points
        radii = [center.distance_to(p) for p in segment]
        avg_radius = sum(radii) / len(radii)

        if avg_radius < 2.0:  # Filter out tiny arcs
            return None

        # Check radius consistency across all points (global fit quality)
        max_deviation = max(abs(r - avg_radius) for r in radii)
        relative_deviation = max_deviation / avg_radius

        if relative_deviation > self.radius_tolerance:
            # Poor fit - try to find best contiguous sub-segment
            return self._try_fit_arc(segment)

        # Good fit! Calculate arc parameters
        start_angle = self._calculate_angle(center, segment[0])
        end_angle = self._calculate_angle(center, segment[-1])

        # Check if it's a full circle
        is_full_circle = False
        if len(segment) >= 8:
            first_to_last_dist = segment[0].distance_to(segment[-1])
            avg_segment_length = sum(segment[i].distance_to(segment[i+1])
                                    for i in range(len(segment)-1)) / (len(segment)-1)

            if first_to_last_dist < 1.2 * avg_segment_length:
                angle_span = self._calculate_angle_span(start_angle, end_angle)
                if angle_span >= 358:
                    is_full_circle = True

        return Arc(
            center=center,
            radius=avg_radius,
            start_angle=start_angle,
            end_angle=end_angle,
            points=segment,
            is_full_circle=is_full_circle
        )

    def detect_arcs_legacy(self, points: List[Tuple[float, float]]) -> List[Arc]:
        """
        Legacy arc detection method (sliding window approach)
        Kept for compatibility/fallback
        """
        if len(points) < self.min_arc_points:
            return []

        points = [Point(p[0], p[1]) for p in points]
        arcs = []
        i = 0

        while i < len(points) - self.min_arc_points + 1:
            arc = self._try_fit_arc(points[i:])
            if arc and len(arc.points) >= self.min_arc_points:
                # Filter out tiny arcs (likely noise or very small features)
                # Minimum radius of 2.0 units to avoid detecting micro-circles
                if arc.radius >= 2.0:
                    arcs.append(arc)
                    # Advance by the full arc length to prevent overlapping detections
                    i += len(arc.points)
                else:
                    i += 1
            else:
                i += 1

        return arcs

    def _try_fit_arc(self, points: List[Point], max_points: int = None) -> Optional[Arc]:
        """
        Try to fit an arc starting from the beginning of points
        Returns Arc if successful, None otherwise
        """
        if max_points is None:
            max_points = len(points)

        # Need at least 3 points to define an arc
        if len(points) < 3:
            return None

        # Check if first 3 points are collinear (not an arc)
        if self._are_collinear(points[0], points[1], points[2]):
            return None

        # Find circle through first 3 points
        center = self._find_circle_center(points[0], points[1], points[2])
        if center is None:
            return None

        radius = center.distance_to(points[0])

        # Check for degenerate case (zero or very small radius)
        if radius < 1e-6:
            return None

        # Try to extend the arc as far as possible
        arc_points = [points[0], points[1], points[2]]

        for i in range(3, min(len(points), max_points)):
            # Check if this point lies on the circle
            dist = center.distance_to(points[i])
            relative_error = abs(dist - radius) / radius

            if relative_error > self.radius_tolerance:
                break

            # Check angular consistency (points should be evenly spaced)
            if not self._check_angular_consistency(center, arc_points[-2:] + [points[i]]):
                break

            arc_points.append(points[i])

        if len(arc_points) < self.min_arc_points:
            return None

        # Calculate arc parameters
        start_angle = self._calculate_angle(center, arc_points[0])
        end_angle = self._calculate_angle(center, arc_points[-1])

        # Check if it's a full circle (end near start)
        is_full_circle = False
        if len(arc_points) >= 8:  # At least 8 points for a circle
            first_to_last_dist = arc_points[0].distance_to(arc_points[-1])
            avg_segment_length = sum(arc_points[i].distance_to(arc_points[i+1])
                                    for i in range(len(arc_points)-1)) / (len(arc_points)-1)

            # If first and last points are very close (within 1.2 segment lengths)
            # Use stricter threshold to avoid misclassifying partial arcs
            if first_to_last_dist < 1.2 * avg_segment_length:
                # Check if arc spans very close to 360 degrees
                # Must be at least 358 degrees to be considered a full circle
                angle_span = self._calculate_angle_span(start_angle, end_angle)
                if angle_span >= 358:  # Very strict: must cover almost full circle
                    is_full_circle = True

        return Arc(
            center=center,
            radius=radius,
            start_angle=start_angle,
            end_angle=end_angle,
            points=arc_points,
            is_full_circle=is_full_circle
        )

    def _are_collinear(self, p1: Point, p2: Point, p3: Point) -> bool:
        """Check if three points are collinear using cross product"""
        # Cross product of vectors (p2-p1) and (p3-p1)
        cross = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)

        # Calculate area of triangle formed by the points
        area = abs(cross) / 2.0

        # Calculate perimeter
        d12 = p1.distance_to(p2)
        d23 = p2.distance_to(p3)
        d31 = p3.distance_to(p1)
        perimeter = d12 + d23 + d31

        if perimeter == 0:
            return True

        # Normalized measure: ratio of area to perimeter squared
        normalized_area = area / (perimeter ** 2)

        return normalized_area < self.collinearity_threshold

    def _find_circle_center(self, p1: Point, p2: Point, p3: Point) -> Optional[Point]:
        """
        Find center of circle passing through three points
        Uses perpendicular bisector method
        """
        # Calculate midpoints
        mid12 = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
        mid23 = Point((p2.x + p3.x) / 2, (p2.y + p3.y) / 2)

        # Calculate slopes of chords
        dx12 = p2.x - p1.x
        dy12 = p2.y - p1.y
        dx23 = p3.x - p2.x
        dy23 = p3.y - p2.y

        # Check for degenerate cases (duplicate points)
        if abs(dx12) < 1e-10 and abs(dy12) < 1e-10:
            return None  # p1 and p2 are the same
        if abs(dx23) < 1e-10 and abs(dy23) < 1e-10:
            return None  # p2 and p3 are the same

        # Handle vertical chords
        if abs(dx12) < 1e-10:
            if abs(dx23) < 1e-10:
                return None  # Both vertical, collinear
            # First chord vertical, perpendicular is horizontal
            cx = mid12.x
            # Perpendicular slope of second chord
            if abs(dy23) < 1e-10:
                return None  # Second chord is horizontal, collinear with perpendicular
            slope23_perp = -dx23 / dy23
            cy = mid23.y + slope23_perp * (cx - mid23.x)
            return Point(cx, cy)

        if abs(dx23) < 1e-10:
            # Second chord vertical, perpendicular is horizontal
            cx = mid23.x
            # Perpendicular slope of first chord
            if abs(dy12) < 1e-10:
                return None  # First chord is horizontal, collinear with perpendicular
            slope12_perp = -dx12 / dy12
            cy = mid12.y + slope12_perp * (cx - mid12.x)
            return Point(cx, cy)

        # Check for horizontal chords
        if abs(dy12) < 1e-10:
            # First chord is horizontal, perpendicular is vertical
            if abs(dy23) < 1e-10:
                return None  # Both horizontal, collinear
            cy = mid12.y
            slope23_perp = -dx23 / dy23
            cx = mid23.x + (cy - mid23.y) / slope23_perp
            return Point(cx, cy)

        if abs(dy23) < 1e-10:
            # Second chord is horizontal, perpendicular is vertical
            cy = mid23.y
            slope12_perp = -dx12 / dy12
            cx = mid12.x + (cy - mid12.y) / slope12_perp
            return Point(cx, cy)

        # Calculate perpendicular slopes
        slope12_perp = -dx12 / dy12
        slope23_perp = -dx23 / dy23

        # Check if perpendiculars are parallel (points are collinear)
        if abs(slope12_perp - slope23_perp) < 1e-10:
            return None

        # Find intersection of perpendicular bisectors
        # Line 1: y - mid12.y = slope12_perp * (x - mid12.x)
        # Line 2: y - mid23.y = slope23_perp * (x - mid23.x)

        cx = ((mid23.y - mid12.y) + slope12_perp * mid12.x - slope23_perp * mid23.x) / \
             (slope12_perp - slope23_perp)
        cy = mid12.y + slope12_perp * (cx - mid12.x)

        return Point(cx, cy)

    def _calculate_angle(self, center: Point, point: Point) -> float:
        """Calculate angle from center to point in degrees [0, 360)"""
        angle = math.degrees(math.atan2(point.y - center.y, point.x - center.x))
        return angle % 360

    def _calculate_angle_span(self, start_angle: float, end_angle: float) -> float:
        """Calculate the angular span of an arc"""
        diff = (end_angle - start_angle) % 360
        return diff

    def _check_angular_consistency(self, center: Point, points: List[Point]) -> bool:
        """
        Check if points are angularly consistent (roughly evenly spaced on arc)
        """
        if len(points) < 2:
            return True

        angles = [self._calculate_angle(center, p) for p in points]

        # Calculate angular differences
        angle_diffs = []
        for i in range(len(angles) - 1):
            diff = (angles[i+1] - angles[i]) % 360
            # Handle wraparound (e.g., 359° to 1°)
            if diff > 180:
                diff = diff - 360
            angle_diffs.append(abs(diff))

        if not angle_diffs:
            return True

        # Check if angular differences are consistent
        avg_diff = sum(angle_diffs) / len(angle_diffs)

        if avg_diff < 0.1:  # Too small, might be noise
            return False

        for diff in angle_diffs:
            if abs(diff - avg_diff) > self.angle_tolerance:
                return False

        return True

    def is_closed_loop(self, points: List[Tuple[float, float]],
                       tolerance_factor: float = 1.5) -> bool:
        """
        Check if polyline forms a closed loop (first point ≈ last point)

        Args:
            points: List of (x, y) points
            tolerance_factor: Multiplier for average segment length to determine closure

        Returns:
            True if first and last points are close enough
        """
        if len(points) < self.min_arc_points:
            return False

        # Calculate average segment length
        total_length = 0
        for i in range(len(points) - 1):
            dx = points[i+1][0] - points[i][0]
            dy = points[i+1][1] - points[i][1]
            total_length += math.sqrt(dx**2 + dy**2)

        avg_segment_length = total_length / (len(points) - 1)

        # Check distance between first and last point
        dx = points[-1][0] - points[0][0]
        dy = points[-1][1] - points[0][1]
        closure_dist = math.sqrt(dx**2 + dy**2)

        return closure_dist < tolerance_factor * avg_segment_length

    def check_radius_consistency(self, points: List[Tuple[float, float]]) -> Optional[Tuple[Point, float, float]]:
        """
        Check if points lie on a circle with consistent radius

        Args:
            points: List of (x, y) points

        Returns:
            (center, radius, relative_deviation) if consistent, None otherwise
        """
        if len(points) < 3:
            return None

        points_obj = [Point(p[0], p[1]) for p in points]

        # Calculate approximate center (centroid)
        cx = sum(p.x for p in points_obj) / len(points_obj)
        cy = sum(p.y for p in points_obj) / len(points_obj)
        center = Point(cx, cy)

        # Calculate distances from center
        radii = [center.distance_to(p) for p in points_obj]
        avg_radius = sum(radii) / len(radii)

        # Check consistency
        max_deviation = max(abs(r - avg_radius) for r in radii)
        relative_deviation = max_deviation / avg_radius if avg_radius > 0 else float('inf')

        return center, avg_radius, relative_deviation

    def detect_circle_global(self, points: List[Tuple[float, float]]) -> Optional[Arc]:
        """
        Detect if polyline is a complete circle using global analysis

        This is a fast preprocessing step for obvious circles before AASR.

        Args:
            points: List of (x, y) points

        Returns:
            Arc object if circle detected, None otherwise
        """
        if len(points) < self.min_arc_points:
            return None

        # Check if it's a closed loop
        if not self.is_closed_loop(points):
            return None

        # Check radius consistency
        result = self.check_radius_consistency(points)
        if result is None:
            return None

        center, radius, rel_deviation = result

        # Must have consistent radius
        if rel_deviation > self.radius_tolerance:
            return None

        # Filter out tiny circles (likely noise)
        if radius < 2.0:
            return None

        # Create full circle arc
        points_obj = [Point(p[0], p[1]) for p in points]
        return Arc(
            center=center,
            radius=radius,
            start_angle=0,
            end_angle=360,
            points=points_obj,
            is_full_circle=True
        )

    def classify_arc(self, arc: Arc) -> str:
        """Classify arc type: full_circle, major_arc, minor_arc, semicircle"""
        if arc.is_full_circle:
            return "full_circle"

        angle_span = self._calculate_angle_span(arc.start_angle, arc.end_angle)

        if abs(angle_span - 180) < 5:
            return "semicircle"
        elif angle_span > 180:
            return "major_arc"
        else:
            return "minor_arc"


def extract_polylines_from_paths(paths: List[Dict[str, Any]]) -> List[List[Tuple[float, float]]]:
    """
    Extract polylines (sequences of connected line segments) from PDF paths
    """
    polylines = []

    for path in paths:
        items = path.get('items', [])
        if not items:
            continue

        current_polyline = []

        for item in items:
            if item['type'] == 'line':
                from_point = item['from']
                to_point = item['to']

                # Parse point strings like "Point(x, y)"
                if isinstance(from_point, str):
                    from_point = parse_point_string(from_point)
                if isinstance(to_point, str):
                    to_point = parse_point_string(to_point)

                if not current_polyline:
                    current_polyline.append(from_point)
                current_polyline.append(to_point)

        if len(current_polyline) >= 3:  # At least 3 points for potential arc
            polylines.append(current_polyline)

    return polylines


def parse_point_string(point_str: str) -> Tuple[float, float]:
    """Parse 'Point(x, y)' string to (x, y) tuple"""
    if isinstance(point_str, (list, tuple)):
        return tuple(point_str[:2])

    # Remove 'Point(' prefix and ')' suffix
    point_str = point_str.replace('Point(', '').replace(')', '')
    parts = point_str.split(',')
    return (float(parts[0].strip()), float(parts[1].strip()))


if __name__ == "__main__":
    # Test with sample circle/arc data
    print("Testing Arc Detection Algorithm")
    print("=" * 80)

    # Test 1: Perfect circle (24 points)
    print("\nTest 1: Perfect Circle (24 points)")
    center_x, center_y = 100, 100
    radius = 50
    circle_points = []
    for i in range(24):
        angle = (i / 24) * 2 * math.pi
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        circle_points.append((x, y))

    detector = ArcDetector(angle_tolerance=10.0, radius_tolerance=0.05, min_arc_points=5)
    arcs = detector.detect_arcs(circle_points)

    print(f"Detected {len(arcs)} arc(s)")
    for i, arc in enumerate(arcs):
        print(f"  Arc {i+1}:")
        print(f"    Center: ({arc.center.x:.2f}, {arc.center.y:.2f})")
        print(f"    Radius: {arc.radius:.2f}")
        print(f"    Angles: {arc.start_angle:.1f}° to {arc.end_angle:.1f}°")
        print(f"    Points: {len(arc.points)}")
        print(f"    Type: {detector.classify_arc(arc)}")
        print(f"    Arc Length: {arc.arc_length():.2f}")

    # Test 2: 90-degree arc
    print("\nTest 2: 90-degree Arc")
    arc_points = []
    for i in range(7):
        angle = (i / 6) * (math.pi / 2)  # 0 to 90 degrees
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        arc_points.append((x, y))

    arcs = detector.detect_arcs(arc_points)
    print(f"Detected {len(arcs)} arc(s)")
    for i, arc in enumerate(arcs):
        print(f"  Arc {i+1}:")
        print(f"    Center: ({arc.center.x:.2f}, {arc.center.y:.2f})")
        print(f"    Radius: {arc.radius:.2f}")
        print(f"    Angles: {arc.start_angle:.1f}° to {arc.end_angle:.1f}°")
        print(f"    Type: {detector.classify_arc(arc)}")

    # Test 3: Straight line (should not detect arc)
    print("\nTest 3: Straight Line (should not detect)")
    line_points = [(0, 0), (10, 10), (20, 20), (30, 30)]
    arcs = detector.detect_arcs(line_points)
    print(f"Detected {len(arcs)} arc(s) - Expected: 0")
