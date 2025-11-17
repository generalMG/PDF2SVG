#!/usr/bin/env python3
"""
Diagnostic test to see what's happening during smoothing and arc detection
"""

import math
from arc_detector import ArcDetector, Point

def generate_zigzag_arc(center_x, center_y, radius, num_points=12, zigzag_amplitude=8):
    """Generate a quarter circle arc with zigzag"""
    points = []
    for i in range(num_points):
        angle = (i / (num_points - 1)) * (math.pi / 2)  # 0 to 90 degrees

        # Add alternating zigzag
        zigzag = zigzag_amplitude * (1 if i % 2 == 0 else -1)

        x = center_x + (radius + zigzag) * math.cos(angle)
        y = center_y + (radius + zigzag) * math.sin(angle)
        points.append((x, y))

    return points

def main():
    print("Diagnostic Test: Zigzag Arc Detection")
    print("=" * 80)

    # Create zigzag quarter arc
    zigzag_points = generate_zigzag_arc(100, 100, 50, 12, 8)

    print(f"\nOriginal zigzag points ({len(zigzag_points)} points):")
    for i, (x, y) in enumerate(zigzag_points):
        print(f"  {i}: ({x:.2f}, {y:.2f})")

    # Create detector
    detector = ArcDetector(
        angle_tolerance=5.0,
        radius_tolerance=0.15,  # More permissive
        min_arc_points=4,
        enable_smoothing=True,
        smoothing_window=5
    )

    # Convert to Point objects
    points_obj = [Point(p[0], p[1]) for p in zigzag_points]

    # Check zigzag detection
    is_zigzag = detector._detect_zigzag_pattern(points_obj)
    print(f"\nZigzag detected: {is_zigzag}")

    # Apply smoothing
    if is_zigzag:
        smoothed_points = detector._smooth_polyline(points_obj)
        print(f"\nSmoothed points ({len(smoothed_points)} points):")
        for i, p in enumerate(smoothed_points):
            print(f"  {i}: ({p.x:.2f}, {p.y:.2f})")

        # Check radius deviation before and after
        center = Point(100, 100)

        print("\nRadius analysis BEFORE smoothing:")
        radii_before = [center.distance_to(p) for p in points_obj]
        avg_r_before = sum(radii_before) / len(radii_before)
        max_dev_before = max(abs(r - avg_r_before) for r in radii_before)
        rel_dev_before = max_dev_before / avg_r_before
        print(f"  Average radius: {avg_r_before:.2f}")
        print(f"  Max deviation: {max_dev_before:.2f}")
        print(f"  Relative deviation: {rel_dev_before:.3f} ({rel_dev_before*100:.1f}%)")

        print("\nRadius analysis AFTER smoothing:")
        radii_after = [center.distance_to(p) for p in smoothed_points]
        avg_r_after = sum(radii_after) / len(radii_after)
        max_dev_after = max(abs(r - avg_r_after) for r in radii_after)
        rel_dev_after = max_dev_after / avg_r_after
        print(f"  Average radius: {avg_r_after:.2f}")
        print(f"  Max deviation: {max_dev_after:.2f}")
        print(f"  Relative deviation: {rel_dev_after:.3f} ({rel_dev_after*100:.1f}%)")

        print(f"\nImprovement: {(rel_dev_before - rel_dev_after) / rel_dev_before * 100:.1f}%")

    # Try arc detection
    print("\n" + "=" * 80)
    print("Arc Detection Results:")
    print("=" * 80)

    arcs = detector.detect_arcs(zigzag_points)
    print(f"\nDetected {len(arcs)} arc(s)")

    if arcs:
        for i, arc in enumerate(arcs):
            angle_span = (arc.end_angle - arc.start_angle) % 360
            print(f"\nArc {i+1}:")
            print(f"  Center: ({arc.center.x:.2f}, {arc.center.y:.2f})")
            print(f"  Radius: {arc.radius:.2f}")
            print(f"  Start angle: {arc.start_angle:.1f}°")
            print(f"  End angle: {arc.end_angle:.1f}°")
            print(f"  Angle span: {angle_span:.1f}°")
            print(f"  Points: {len(arc.points)}")
    else:
        print("\nNo arcs detected. Trying with more permissive settings...")

        # Try with very permissive settings
        detector2 = ArcDetector(
            angle_tolerance=10.0,
            radius_tolerance=0.20,
            min_arc_points=4,
            enable_smoothing=True,
            smoothing_window=7  # Even larger window
        )

        arcs2 = detector2.detect_arcs(zigzag_points)
        print(f"With permissive settings: {len(arcs2)} arc(s) detected")

        if arcs2:
            for i, arc in enumerate(arcs2):
                angle_span = (arc.end_angle - arc.start_angle) % 360
                print(f"\nArc {i+1}:")
                print(f"  Center: ({arc.center.x:.2f}, {arc.center.y:.2f})")
                print(f"  Radius: {arc.radius:.2f}")
                print(f"  Angle span: {angle_span:.1f}°")
                print(f"  Points: {len(arc.points)}")

if __name__ == "__main__":
    main()
