#!/usr/bin/env python3
"""
Test circle merging logic
"""
import math
from arc_detector import ArcDetector, Arc, Point

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
print("TESTING CIRCLE MERGING LOGIC")
print("=" * 80)

detector = ArcDetector(angle_tolerance=10.0, radius_tolerance=0.05, min_arc_points=8)

# Simulate a large circle split into 4 arcs (like a ball bearing outer ring)
# Center at (100, 100), radius 50
center = (100, 100)
radius = 50

# Create 4 separate arcs that together form a full circle
arc1_points = create_arc_points(100, 100, 50, 0, 90, 15)     # 0-90 degrees
arc2_points = create_arc_points(100, 100, 50, 90, 180, 15)   # 90-180 degrees
arc3_points = create_arc_points(100, 100, 50, 180, 270, 15)  # 180-270 degrees
arc4_points = create_arc_points(100, 100, 50, 270, 360, 15)  # 270-360 degrees

# Detect arcs separately (simulating multiple polylines in PDF)
detected_arcs = []

for i, arc_pts in enumerate([arc1_points, arc2_points, arc3_points, arc4_points], 1):
    arcs = detector.detect_arcs(arc_pts)
    if arcs:
        arc = arcs[0]
        detected_arcs.append({
            'arc': arc,
            'is_circle': False,
            'center': arc.center.to_tuple(),
            'radius': arc.radius,
            'stroke': 'black',
            'fill': 'none',
            'stroke_width': 1.0,
            'segments_saved': len(arc.points) - 1
        })
        print(f"\nArc {i}:")
        print(f"  Center: ({arc.center.x:.2f}, {arc.center.y:.2f})")
        print(f"  Radius: {arc.radius:.2f}")
        print(f"  Angle: {arc.start_angle:.1f}° to {arc.end_angle:.1f}°")
        print(f"  Points: {len(arc.points)}")

print(f"\n{'='*80}")
print(f"Total detected arcs: {len(detected_arcs)}")

# Test the merging logic (simulate what pdf_to_svg.py does)
print("\n" + "=" * 80)
print("TESTING MERGE LOGIC")
print("=" * 80)

# Group arcs by center and radius
arc_groups = {}
for arc_info in detected_arcs:
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

print(f"\nArc groups found: {len(arc_groups)}")

merged_circles = []
remaining_arcs = []

# Check each group to see if arcs combine into a circle
for (cx, cy, r), group_arcs in arc_groups.items():
    print(f"\nGroup: center=({cx:.1f}, {cy:.1f}), radius={r:.1f}")
    print(f"  Contains {len(group_arcs)} arc(s)")

    # Calculate total angle coverage
    total_angle_coverage = 0
    for arc_info in group_arcs:
        arc = arc_info['arc']
        angle_span = (arc.end_angle - arc.start_angle) % 360
        total_angle_coverage += angle_span
        print(f"    Arc: {arc.start_angle:.1f}° to {arc.end_angle:.1f}° (span: {angle_span:.1f}°)")

    print(f"  Total angle coverage: {total_angle_coverage:.1f}°")

    # If multiple arcs cover close to 360 degrees, merge into circle
    if len(group_arcs) >= 2 and total_angle_coverage >= 340:
        print(f"  ✓ MERGING into single circle!")
        merged_circles.append({
            'center': (cx, cy),
            'radius': r
        })
    else:
        print(f"  ✗ Keeping as individual arcs")
        remaining_arcs.extend(group_arcs)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Merged circles: {len(merged_circles)}")
print(f"Remaining arcs: {len(remaining_arcs)}")

if merged_circles:
    print("\n✓ SUCCESS: Multiple arcs were merged into a complete circle!")
    print("This should fix the 'many circles' issue for ball bearing rings.")
else:
    print("\n✗ FAILURE: No arcs were merged")
