#!/usr/bin/env python3
"""
Enhanced Validation Metrics for Visualization Tests

Provides detailed point coverage, arc span validation, and overlay metrics
to ensure detected arcs actually contain the input points.
"""

import math
from typing import List, Tuple, Dict, Optional
from arc_detector import Arc, Point


def calculate_point_coverage(input_points: List[Tuple[float, float]],
                             detected_arcs: List[Arc],
                             radius_tolerance: float = 0.02,
                             angle_tolerance: float = 5.0) -> Dict:
    """
    Calculate what percentage of input points are covered by detected arcs.

    This validates that detected arcs actually overlay the input points,
    not the empty complementary arc.

    Args:
        input_points: Original input points
        detected_arcs: Detected Arc objects
        radius_tolerance: Tolerance for radius matching (default: 2%)
        angle_tolerance: Tolerance for angle matching in degrees (default: 5°)

    Returns:
        Dictionary with coverage metrics:
        {
            'total_points': int,
            'covered_points': int,
            'coverage_percent': float,
            'uncovered_points': List[Tuple],
            'coverage_by_arc': List[int],  # Points covered by each arc
            'avg_distance_to_arc': float,
            'max_distance_to_arc': float
        }
    """
    if not detected_arcs:
        return {
            'total_points': len(input_points),
            'covered_points': 0,
            'coverage_percent': 0.0,
            'uncovered_points': input_points.copy(),
            'coverage_by_arc': [],
            'avg_distance_to_arc': float('inf'),
            'max_distance_to_arc': float('inf')
        }

    covered_points = 0
    uncovered_points = []
    coverage_by_arc = [0] * len(detected_arcs)
    distances = []

    for point in input_points:
        px, py = point
        point_covered = False
        min_dist = float('inf')

        for arc_idx, arc in enumerate(detected_arcs):
            # Calculate distance from point to arc center
            dist_to_center = math.sqrt((px - arc.center.x)**2 + (py - arc.center.y)**2)

            # Check if point is on the radius (within tolerance)
            radius_error = abs(dist_to_center - arc.radius) / arc.radius
            on_radius = radius_error < radius_tolerance

            if on_radius:
                # Calculate angle of point relative to arc center
                angle = calculate_angle(arc.center, Point(px, py))

                # Check if angle is within arc span (with tolerance)
                if angle_in_arc_range(angle, arc.start_angle, arc.end_angle, angle_tolerance):
                    point_covered = True
                    coverage_by_arc[arc_idx] += 1
                    min_dist = min(min_dist, abs(dist_to_center - arc.radius))
                    break
            else:
                # Track distance even if not on arc
                min_dist = min(min_dist, abs(dist_to_center - arc.radius))

        if point_covered:
            covered_points += 1
            distances.append(min_dist)
        else:
            uncovered_points.append(point)
            if min_dist != float('inf'):
                distances.append(min_dist)

    coverage_percent = (covered_points / len(input_points) * 100) if input_points else 0.0
    avg_distance = sum(distances) / len(distances) if distances else float('inf')
    max_distance = max(distances) if distances else float('inf')

    return {
        'total_points': len(input_points),
        'covered_points': covered_points,
        'coverage_percent': coverage_percent,
        'uncovered_points': uncovered_points,
        'coverage_by_arc': coverage_by_arc,
        'avg_distance_to_arc': avg_distance,
        'max_distance_to_arc': max_distance
    }


def validate_arc_span(detected_arc: Arc,
                      expected_span: float,
                      tolerance: float = 10.0) -> Tuple[bool, float, str]:
    """
    Validate that detected arc span matches expected span.

    Args:
        detected_arc: Detected Arc object
        expected_span: Expected arc span in degrees
        tolerance: Tolerance in degrees (default: 10°)

    Returns:
        Tuple of (is_valid, actual_span, message)
    """
    # Calculate actual arc span
    start = detected_arc.start_angle % 360
    end = detected_arc.end_angle % 360

    if start <= end:
        actual_span = end - start
    else:
        # Wraparound case
        actual_span = 360 - start + end

    # Check difference
    diff = abs(actual_span - expected_span)
    is_valid = diff < tolerance

    if is_valid:
        message = f"✓ Arc span {actual_span:.1f}° matches expected {expected_span:.1f}° (±{diff:.1f}°)"
    else:
        message = f"✗ Arc span {actual_span:.1f}° differs from expected {expected_span:.1f}° by {diff:.1f}°"

    return is_valid, actual_span, message


def check_arc_direction(input_points: List[Tuple[float, float]],
                       detected_arc: Arc,
                       expected_span: float) -> Tuple[bool, str]:
    """
    Check if arc is detected on correct side (not complementary arc).

    For a 270° arc, this ensures we detect the 270° segment with points,
    not the empty 90° complement.

    Args:
        input_points: Original input points
        detected_arc: Detected Arc object
        expected_span: Expected arc span in degrees

    Returns:
        Tuple of (is_correct_side, message)
    """
    # Calculate angles for all input points
    point_angles = []
    for px, py in input_points:
        angle = calculate_angle(detected_arc.center, Point(px, py))
        point_angles.append(angle)

    # Count points inside detected arc span
    points_in_arc = 0
    points_outside_arc = 0

    start = detected_arc.start_angle % 360
    end = detected_arc.end_angle % 360

    for angle in point_angles:
        if angle_in_range(angle, start, end):
            points_in_arc += 1
        else:
            points_outside_arc += 1

    # Calculate detected span
    if start <= end:
        detected_span = end - start
    else:
        detected_span = 360 - start + end

    # Points should be on the detected arc, not the complement
    on_correct_side = points_in_arc > points_outside_arc

    # Additional check: if expected span is large (>180°), detected span should also be large
    span_matches = abs(detected_span - expected_span) < 45  # Within 45° of expected

    if on_correct_side and span_matches:
        message = f"✓ Arc on correct side: {points_in_arc}/{len(point_angles)} points in arc (span: {detected_span:.1f}°)"
    elif not on_correct_side:
        message = f"✗ Arc on WRONG side: {points_in_arc}/{len(point_angles)} points in arc, {points_outside_arc} outside (COMPLEMENTARY ARC DETECTED)"
        message += f"\n  Detected span: {detected_span:.1f}°, Expected: {expected_span:.1f}°"
        message += f"\n  FIX: Arc should be on opposite side (swap start/end angles)"
    else:
        message = f"⚠ Arc span mismatch: detected {detected_span:.1f}°, expected {expected_span:.1f}° (but points are on correct side)"

    return on_correct_side and span_matches, message


def analyze_detection_quality(input_points: List[Tuple[float, float]],
                              detected_arcs: List[Arc],
                              expected_arcs: int,
                              expected_span: Optional[float] = None) -> Dict:
    """
    Comprehensive analysis of detection quality.

    Args:
        input_points: Original input points
        detected_arcs: List of detected Arc objects
        expected_arcs: Expected number of arcs
        expected_span: Expected arc span in degrees (for single arc tests)

    Returns:
        Dictionary with comprehensive metrics
    """
    # Count check
    count_match = len(detected_arcs) == expected_arcs

    # Coverage metrics
    coverage = calculate_point_coverage(input_points, detected_arcs)

    # Arc span validation (for single arc tests)
    span_valid = True
    span_message = ""
    direction_valid = True
    direction_message = ""

    if expected_span is not None and len(detected_arcs) == 1:
        span_valid, actual_span, span_message = validate_arc_span(
            detected_arcs[0], expected_span
        )
        direction_valid, direction_message = check_arc_direction(
            input_points, detected_arcs[0], expected_span
        )

    # Overall pass/fail
    overall_pass = (
        count_match and
        coverage['coverage_percent'] >= 95.0 and  # At least 95% coverage
        span_valid and
        direction_valid
    )

    return {
        'count_match': count_match,
        'coverage': coverage,
        'span_valid': span_valid,
        'span_message': span_message,
        'direction_valid': direction_valid,
        'direction_message': direction_message,
        'overall_pass': overall_pass
    }


def format_test_result(test_name: str,
                      analysis: Dict,
                      expected_arcs: int,
                      detected_arcs: List[Arc]) -> str:
    """
    Format test results with enhanced metrics.

    Args:
        test_name: Name of the test
        analysis: Analysis dictionary from analyze_detection_quality
        expected_arcs: Expected number of arcs
        detected_arcs: Detected Arc objects

    Returns:
        Formatted test result string
    """
    lines = []
    lines.append("-" * 80)
    lines.append(f"TEST: {test_name}")
    lines.append("-" * 80)

    # Basic count
    lines.append(f"EXPECTED: {expected_arcs} arc(s)")
    lines.append(f"ACTUAL:   {len(detected_arcs)} arc(s)")

    # Coverage metrics
    cov = analysis['coverage']
    lines.append(f"COVERAGE: {cov['coverage_percent']:.1f}% ({cov['covered_points']}/{cov['total_points']} points)")

    if cov['coverage_by_arc']:
        for i, count in enumerate(cov['coverage_by_arc']):
            lines.append(f"  Arc {i+1}: {count} points")

    lines.append(f"AVG DIST: {cov['avg_distance_to_arc']:.4f} units")
    lines.append(f"MAX DIST: {cov['max_distance_to_arc']:.4f} units")

    # Arc span validation
    if analysis['span_message']:
        lines.append(analysis['span_message'])

    # Direction validation
    if analysis['direction_message']:
        for msg_line in analysis['direction_message'].split('\n'):
            lines.append(msg_line)

    # Detailed arc info
    if detected_arcs:
        lines.append("")
        lines.append("DETECTED ARCS:")
        for i, arc in enumerate(detected_arcs):
            span = calculate_arc_span(arc.start_angle, arc.end_angle)
            lines.append(f"  Arc {i+1}:")
            lines.append(f"    Center: ({arc.center.x:.2f}, {arc.center.y:.2f})")
            lines.append(f"    Radius: {arc.radius:.2f}")
            lines.append(f"    Span: {arc.start_angle:.1f}° → {arc.end_angle:.1f}° ({span:.1f}°)")
            lines.append(f"    Points: {len(arc.points)}")

    # Overall status
    lines.append("")
    if analysis['overall_pass']:
        lines.append("STATUS:   ✓ PASS")
    else:
        lines.append("STATUS:   ✗ FAIL")
        if not analysis['count_match']:
            lines.append("  - Arc count mismatch")
        if analysis['coverage']['coverage_percent'] < 95.0:
            lines.append(f"  - Low coverage ({analysis['coverage']['coverage_percent']:.1f}%)")
        if not analysis['span_valid']:
            lines.append("  - Arc span mismatch")
        if not analysis['direction_valid']:
            lines.append("  - Arc on wrong side (complementary arc)")

    lines.append("-" * 80)
    lines.append("")

    return '\n'.join(lines)


# Helper functions

def calculate_angle(center: Point, point: Point) -> float:
    """Calculate angle in degrees from center to point (0-360)."""
    dx = point.x - center.x
    dy = point.y - center.y
    angle = math.degrees(math.atan2(dy, dx))
    return angle % 360


def calculate_arc_span(start_angle: float, end_angle: float) -> float:
    """Calculate arc span handling wraparound."""
    start = start_angle % 360
    end = end_angle % 360
    if start <= end:
        return end - start
    else:
        return 360 - start + end


def angle_in_range(angle: float, start: float, end: float) -> bool:
    """Check if angle is in range from start to end (counterclockwise)."""
    angle = angle % 360
    start = start % 360
    end = end % 360

    if start <= end:
        return start <= angle <= end
    else:
        # Wraparound case
        return angle >= start or angle <= end


def angle_in_arc_range(angle: float, start: float, end: float, tolerance: float = 0) -> bool:
    """
    Check if angle is within arc range with tolerance.

    Args:
        angle: Angle to check
        start: Arc start angle
        end: Arc end angle
        tolerance: Tolerance in degrees

    Returns:
        True if angle is in range (with tolerance)
    """
    # Normalize angles
    angle = angle % 360
    start = (start - tolerance) % 360
    end = (end + tolerance) % 360

    if start <= end:
        return start <= angle <= end
    else:
        # Wraparound
        return angle >= start or angle <= end


def print_summary(results: List[Tuple[str, Dict]]):
    """
    Print summary of all test results.

    Args:
        results: List of (test_name, analysis) tuples
    """
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    total = len(results)
    passed = sum(1 for _, analysis in results if analysis['overall_pass'])
    failed = total - passed

    print(f"Total Tests:  {total}")
    print(f"Passed:       {passed} ({passed/total*100:.1f}%)")
    print(f"Failed:       {failed} ({failed/total*100:.1f}%)")

    if failed > 0:
        print("")
        print("FAILED TESTS:")
        for name, analysis in results:
            if not analysis['overall_pass']:
                print(f"  - {name}")
                if analysis['coverage']['coverage_percent'] < 95.0:
                    print(f"    Coverage: {analysis['coverage']['coverage_percent']:.1f}%")
                if not analysis['direction_valid']:
                    print(f"    Issue: Arc on wrong side (complementary)")

    print("=" * 80)
