#!/usr/bin/env python3
"""
Diagnostic tool to analyze polyline patterns in PDF
"""

import fitz
import sys
from pathlib import Path
from arc_detector import ArcDetector, Point
import math

def analyze_polyline_angles(points):
    """Analyze angle deviations in a polyline"""
    if len(points) < 3:
        return None

    angle_changes = []
    for i in range(1, len(points) - 1):
        v1_x = points[i][0] - points[i-1][0]
        v1_y = points[i][1] - points[i-1][1]
        v2_x = points[i+1][0] - points[i][0]
        v2_y = points[i+1][1] - points[i][1]

        len_v1 = math.sqrt(v1_x**2 + v1_y**2)
        len_v2 = math.sqrt(v2_x**2 + v2_y**2)

        if len_v1 < 1e-6 or len_v2 < 1e-6:
            continue

        # Normalize
        v1_x, v1_y = v1_x / len_v1, v1_y / len_v1
        v2_x, v2_y = v2_x / len_v2, v2_y / len_v2

        # Angle
        dot = v1_x * v2_x + v1_y * v2_y
        dot = max(-1.0, min(1.0, dot))
        angle = math.degrees(math.acos(dot))

        # Cross product for sign
        cross = v1_x * v2_y - v1_y * v2_x
        signed_angle = angle if cross >= 0 else -angle

        angle_changes.append(signed_angle)

    if not angle_changes:
        return None

    # Calculate statistics
    avg_angle = sum(abs(a) for a in angle_changes) / len(angle_changes)
    max_angle = max(abs(a) for a in angle_changes)
    min_angle = min(abs(a) for a in angle_changes)

    # Check for zigzag pattern
    sign_changes = sum(1 for i in range(len(angle_changes) - 1)
                      if angle_changes[i] * angle_changes[i+1] < 0)
    alternation_ratio = sign_changes / (len(angle_changes) - 1) if len(angle_changes) > 1 else 0

    return {
        'avg_angle': avg_angle,
        'max_angle': max_angle,
        'min_angle': min_angle,
        'alternation_ratio': alternation_ratio,
        'num_angles': len(angle_changes),
        'is_zigzag': alternation_ratio > 0.5 and avg_angle > 0.5
    }

def extract_polylines_from_pdf(pdf_path, page_num=0):
    """Extract polylines from PDF"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    paths = page.get_drawings()

    all_polylines = []

    for path in paths:
        items = path.get('items', [])
        current_polyline = []

        for item in items:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue

            operation = item[0]

            if operation != 'l':
                if len(current_polyline) >= 2:
                    all_polylines.append(current_polyline)
                current_polyline = []
                continue

            # Line segment
            try:
                from_pt = (float(item[1].x), float(item[1].y)) if hasattr(item[1], 'x') else (float(item[1][0]), float(item[1][1]))
                to_pt = (float(item[2].x), float(item[2].y)) if hasattr(item[2], 'x') else (float(item[2][0]), float(item[2][1]))

                if not current_polyline:
                    current_polyline = [from_pt, to_pt]
                else:
                    current_polyline.append(to_pt)
            except:
                continue

        if len(current_polyline) >= 2:
            all_polylines.append(current_polyline)

    doc.close()
    return all_polylines

def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_polylines.py <pdf_file> [page_num]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    page_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    print(f"Analyzing: {pdf_path} (page {page_num})")
    print("=" * 80)

    polylines = extract_polylines_from_pdf(pdf_path, page_num)

    print(f"\nTotal polylines extracted: {len(polylines)}")
    print("\nPolyline Analysis:")
    print("=" * 80)

    # Categorize polylines
    very_short = []  # < 5 points
    short = []       # 5-10 points
    medium = []      # 11-20 points
    long = []        # > 20 points

    zigzag_polylines = []
    subtle_zigzag = []  # avg angle < 2.0 but still zigzag

    for i, polyline in enumerate(polylines):
        num_points = len(polyline)

        if num_points < 5:
            very_short.append((i, polyline))
        elif num_points <= 10:
            short.append((i, polyline))
        elif num_points <= 20:
            medium.append((i, polyline))
        else:
            long.append((i, polyline))

        # Analyze angles
        analysis = analyze_polyline_angles(polyline)
        if analysis and analysis['is_zigzag']:
            zigzag_polylines.append((i, polyline, analysis))
            if analysis['avg_angle'] < 2.0:
                subtle_zigzag.append((i, polyline, analysis))

    print(f"\nPolyline length distribution:")
    print(f"  Very short (< 5 points):  {len(very_short)}")
    print(f"  Short (5-10 points):      {len(short)}")
    print(f"  Medium (11-20 points):    {len(medium)}")
    print(f"  Long (> 20 points):       {len(long)}")

    print(f"\nZigzag analysis:")
    print(f"  Total zigzag polylines:   {len(zigzag_polylines)}")
    print(f"  Subtle zigzags (<2°):     {len(subtle_zigzag)}")

    # Show details of subtle zigzags
    if subtle_zigzag:
        print(f"\nSubtle Zigzag Details (first 5):")
        print("-" * 80)
        for idx, polyline, analysis in subtle_zigzag[:5]:
            print(f"\nPolyline #{idx}:")
            print(f"  Points: {len(polyline)}")
            print(f"  Avg angle: {analysis['avg_angle']:.3f}°")
            print(f"  Max angle: {analysis['max_angle']:.3f}°")
            print(f"  Min angle: {analysis['min_angle']:.3f}°")
            print(f"  Alternation: {analysis['alternation_ratio']:.1%}")
            print(f"  First 3 points: {polyline[:3]}")

    # Show some examples of different length polylines
    print(f"\n\nExample Polylines:")
    print("=" * 80)

    if very_short:
        idx, polyline = very_short[0]
        analysis = analyze_polyline_angles(polyline)
        print(f"\nVery short polyline #{idx} ({len(polyline)} points):")
        print(f"  Points: {polyline}")
        if analysis:
            print(f"  Avg angle: {analysis['avg_angle']:.3f}°")
            print(f"  Zigzag: {analysis['is_zigzag']}")

    if medium:
        idx, polyline = medium[0]
        analysis = analyze_polyline_angles(polyline)
        print(f"\nMedium polyline #{idx} ({len(polyline)} points):")
        print(f"  First 5 points: {polyline[:5]}")
        print(f"  Last 5 points: {polyline[-5:]}")
        if analysis:
            print(f"  Avg angle: {analysis['avg_angle']:.3f}°")
            print(f"  Zigzag: {analysis['is_zigzag']}")

    # Test arc detection
    print(f"\n\nArc Detection Test:")
    print("=" * 80)

    detector = ArcDetector(
        angle_tolerance=5.0,
        radius_tolerance=0.05,
        min_arc_points=4,
        enable_smoothing=True,
        smoothing_window=5
    )

    total_arcs = 0
    for i, polyline in enumerate(polylines):
        if len(polyline) >= 4:
            arcs = detector.detect_arcs(polyline)
            if arcs:
                total_arcs += len(arcs)
                if total_arcs <= 5:  # Show first 5
                    print(f"\nPolyline #{i} ({len(polyline)} points) -> {len(arcs)} arc(s)")
                    for arc in arcs:
                        print(f"  Radius: {arc.radius:.2f}, Points: {len(arc.points)}")

    print(f"\nTotal arcs detected: {total_arcs}")

if __name__ == "__main__":
    main()
