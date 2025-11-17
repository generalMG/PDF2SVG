#!/usr/bin/env python3
"""
Test script for zigzag smoothing algorithm
"""

import math
from arc_detector import ArcDetector, Point

def generate_zigzag_arc(center_x, center_y, radius, num_points=24, zigzag_amplitude=5):
    """
    Generate a circular arc with zigzag noise

    Args:
        center_x, center_y: Circle center
        radius: Circle radius
        num_points: Number of points
        zigzag_amplitude: Amount of zigzag deviation

    Returns:
        List of (x, y) tuples with zigzag pattern
    """
    points = []
    for i in range(num_points):
        angle = (i / num_points) * 2 * math.pi

        # Add zigzag: alternating inward/outward deviation
        zigzag = zigzag_amplitude * (1 if i % 2 == 0 else -1)

        x = center_x + (radius + zigzag) * math.cos(angle)
        y = center_y + (radius + zigzag) * math.sin(angle)
        points.append((x, y))

    return points

def test_zigzag_detection():
    """Test if zigzag pattern is detected correctly"""
    print("=" * 80)
    print("Test 1: Zigzag Detection")
    print("=" * 80)

    # Create zigzag circle
    zigzag_points = generate_zigzag_arc(100, 100, 50, 24, zigzag_amplitude=5)

    # Create smooth circle for comparison
    smooth_points = generate_zigzag_arc(100, 100, 50, 24, zigzag_amplitude=0)

    detector = ArcDetector(
        angle_tolerance=5.0,
        radius_tolerance=0.05,
        min_arc_points=4,
        enable_smoothing=True,
        smoothing_window=5
    )

    # Test zigzag detection
    zigzag_points_obj = [Point(p[0], p[1]) for p in zigzag_points]
    smooth_points_obj = [Point(p[0], p[1]) for p in smooth_points]

    is_zigzag_detected = detector._detect_zigzag_pattern(zigzag_points_obj)
    is_smooth_detected = detector._detect_zigzag_pattern(smooth_points_obj)

    print(f"\nZigzag pattern in noisy circle: {is_zigzag_detected}")
    print(f"Zigzag pattern in smooth circle: {is_smooth_detected}")

    if is_zigzag_detected and not is_smooth_detected:
        print("✓ Zigzag detection working correctly!")
    else:
        print("✗ Zigzag detection needs adjustment")

    return is_zigzag_detected

def test_smoothing_effect():
    """Test if smoothing improves arc detection"""
    print("\n" + "=" * 80)
    print("Test 2: Smoothing Effect on Arc Detection")
    print("=" * 80)

    # Create zigzag quarter circle (90 degrees)
    num_points = 12
    zigzag_arc_points = []
    for i in range(num_points):
        angle = (i / (num_points - 1)) * (math.pi / 2)  # 0 to 90 degrees

        # Add zigzag
        zigzag = 8 * (1 if i % 2 == 0 else -1)

        x = 100 + (50 + zigzag) * math.cos(angle)
        y = 100 + (50 + zigzag) * math.sin(angle)
        zigzag_arc_points.append((x, y))

    # Test without smoothing
    detector_no_smooth = ArcDetector(
        angle_tolerance=5.0,
        radius_tolerance=0.10,
        min_arc_points=4,
        enable_smoothing=False
    )

    # Test with smoothing
    detector_with_smooth = ArcDetector(
        angle_tolerance=5.0,
        radius_tolerance=0.10,
        min_arc_points=4,
        enable_smoothing=True,
        smoothing_window=5
    )

    arcs_no_smooth = detector_no_smooth.detect_arcs(zigzag_arc_points)
    arcs_with_smooth = detector_with_smooth.detect_arcs(zigzag_arc_points)

    print(f"\nZigzag arc points: {num_points}")
    print(f"\nArcs detected WITHOUT smoothing: {len(arcs_no_smooth)}")
    for i, arc in enumerate(arcs_no_smooth):
        angle_span = (arc.end_angle - arc.start_angle) % 360
        print(f"  Arc {i+1}: radius={arc.radius:.2f}, angle_span={angle_span:.1f}°, points={len(arc.points)}")

    print(f"\nArcs detected WITH smoothing: {len(arcs_with_smooth)}")
    for i, arc in enumerate(arcs_with_smooth):
        angle_span = (arc.end_angle - arc.start_angle) % 360
        print(f"  Arc {i+1}: radius={arc.radius:.2f}, angle_span={angle_span:.1f}°, points={len(arc.points)}")

    if len(arcs_with_smooth) > len(arcs_no_smooth):
        print("\n✓ Smoothing improved arc detection!")
    elif len(arcs_with_smooth) == len(arcs_no_smooth) and arcs_with_smooth:
        print("\n✓ Both detected arcs successfully!")
    else:
        print("\n⚠ Check if smoothing parameters need tuning")

def test_straight_line_preservation():
    """Test that straight lines are not affected by smoothing"""
    print("\n" + "=" * 80)
    print("Test 3: Straight Line Preservation")
    print("=" * 80)

    # Create perfect straight line
    straight_line = [(i * 10, i * 10) for i in range(10)]

    # Create horizontal line
    horizontal_line = [(i * 10, 50) for i in range(10)]

    detector = ArcDetector(
        angle_tolerance=5.0,
        radius_tolerance=0.05,
        min_arc_points=4,
        enable_smoothing=True,
        smoothing_window=5
    )

    # Check zigzag detection on straight lines
    straight_points = [Point(p[0], p[1]) for p in straight_line]
    horizontal_points = [Point(p[0], p[1]) for p in horizontal_line]

    is_straight_zigzag = detector._detect_zigzag_pattern(straight_points)
    is_horizontal_zigzag = detector._detect_zigzag_pattern(horizontal_points)

    print(f"\nDiagonal line detected as zigzag: {is_straight_zigzag}")
    print(f"Horizontal line detected as zigzag: {is_horizontal_zigzag}")

    # Try arc detection (should find no arcs)
    arcs_straight = detector.detect_arcs(straight_line)
    arcs_horizontal = detector.detect_arcs(horizontal_line)

    print(f"\nArcs detected in diagonal line: {len(arcs_straight)}")
    print(f"Arcs detected in horizontal line: {len(arcs_horizontal)}")

    if not is_straight_zigzag and not is_horizontal_zigzag and len(arcs_straight) == 0 and len(arcs_horizontal) == 0:
        print("\n✓ Straight lines preserved correctly!")
    else:
        print("\n⚠ Straight line handling may need adjustment")

def main():
    print("\nZigzag Smoothing Algorithm Tests")
    print("=" * 80)

    # Run tests
    test_zigzag_detection()
    test_smoothing_effect()
    test_straight_line_preservation()

    print("\n" + "=" * 80)
    print("Tests complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
