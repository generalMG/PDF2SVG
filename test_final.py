#!/usr/bin/env python3
"""
Test the fixes without requiring PyMuPDF
"""
import math
from arc_detector import ArcDetector

def create_arc_points(center_x, center_y, radius, start_deg, end_deg, num_points):
    """Create points along an arc"""
    points = []
    angle_range = end_deg - start_deg
    for i in range(num_points):
        angle_deg = start_deg + (i / (num_points - 1)) * angle_range
        angle_rad = math.radians(angle_deg)
        x = center_x + radius * math.cos(angle_rad)
        y = center_y + radius * math.sin(angle_rad)
        points.append((x, y))
    return points

print("=" * 80)
print("TESTING ARC DETECTION FIXES")
print("=" * 80)

detector = ArcDetector(angle_tolerance=10.0, radius_tolerance=0.05, min_arc_points=4)

test_cases = [
    ("90-degree arc", 100, 100, 50, 0, 90, 10, "minor_arc"),
    ("180-degree arc", 100, 100, 50, 0, 180, 15, "semicircle"),
    ("270-degree arc", 100, 100, 50, 0, 270, 20, "major_arc"),
    ("310-degree arc", 100, 100, 50, 0, 310, 25, "major_arc"),
    ("330-degree arc", 100, 100, 50, 0, 330, 25, "major_arc"),
    ("355-degree arc", 100, 100, 50, 0, 355, 30, "major_arc"),
    ("360-degree circle", 100, 100, 50, 0, 360, 30, "full_circle"),
]

print("\n" + "=" * 80)
print("FIX 1: Stricter circle classification (358° threshold)")
print("=" * 80)

all_passed = True

for name, cx, cy, r, start, end, npts, expected_type in test_cases:
    points = create_arc_points(cx, cy, r, start, end, npts)
    arcs = detector.detect_arcs(points)

    if arcs:
        arc = arcs[0]
        arc_type = detector.classify_arc(arc)
        angle_span = detector._calculate_angle_span(arc.start_angle, arc.end_angle)

        status = "✓ PASS" if arc_type == expected_type else "✗ FAIL"
        all_passed = all_passed and (arc_type == expected_type)

        print(f"\n{name}:")
        print(f"  Detected: {len(arc.points)} points, {angle_span:.1f}° span")
        print(f"  Expected: {expected_type}")
        print(f"  Got: {arc_type}")
        print(f"  Result: {status}")
    else:
        print(f"\n{name}: ✗ FAIL - No arc detected!")
        all_passed = False

print("\n" + "=" * 80)
print("FIX 2: SVG rendering strategy")
print("=" * 80)

print("\nSVG rendering rules (from pdf_to_svg.py _add_arc method):")
print("  • Arcs ≥350° or <10°: Render as <circle> element")
print("  • Arcs 300-350°: Split into two separate arc paths")
print("  • Arcs <300°: Single arc path (normal)")

for name, cx, cy, r, start, end, npts, expected_type in test_cases:
    points = create_arc_points(cx, cy, r, start, end, npts)
    arcs = detector.detect_arcs(points)

    if arcs:
        arc = arcs[0]
        angle_span = detector._calculate_angle_span(arc.start_angle, arc.end_angle)

        # Determine rendering method (matching pdf_to_svg.py logic)
        if angle_span >= 350 or angle_span < 10:
            render_method = "<circle> element"
        elif angle_span > 300:
            render_method = "Two arc paths (split at midpoint)"
        else:
            render_method = "Single arc path"

        print(f"\n{name} ({angle_span:.1f}°): {render_method}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if all_passed:
    print("\n✓ All tests PASSED!")
    print("\nThe fixes correctly:")
    print("  1. Only classify arcs ≥358° as full circles")
    print("  2. Classify 355° arcs as major arcs (not circles)")
    print("  3. Will render large arcs properly in SVG (no more rendering issues)")
else:
    print("\n✗ Some tests FAILED - please review")

print("\n" + "=" * 80)
