#!/usr/bin/env python3
"""
Visualization of Arc Detection Algorithm (AASR)

Shows the complete AASR pipeline:
1. Input polyline
2. Curvature segmentation (colored by segment)
3. Circle fitting for each segment
4. Validation checks (radius consistency, collinearity)
5. Final detected arcs with parameters

Enhanced with comprehensive validation metrics:
- Point coverage analysis
- Arc direction validation
- Arc span verification
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from arc_detector import ArcDetector, Point, Arc
from visualization_validation import (
    analyze_detection_quality,
    format_test_result,
    print_summary
)

def create_test_paths():
    """Create various test paths for visualization"""
    paths = {}
    expected = {}

    # 1. Simple arc (90 degrees) - SHOULD DETECT 1 arc
    center_x, center_y = 100, 100
    radius = 50
    arc_90 = []
    for i in range(25):
        angle = (i / 24) * (math.pi / 2)  # 0 to 90 degrees
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        arc_90.append((x, y))
    paths['90° Arc'] = arc_90
    expected['90° Arc'] = {
        'should_detect': True,
        'expected_arcs': 1,
        'expected_span': 90.0,
        'description': 'Single 90° arc'
    }

    # 2. Large arc (270 degrees) - SHOULD DETECT 1 arc
    arc_270 = []
    for i in range(75):
        angle = (i / 74) * (3 * math.pi / 2)  # 0 to 270 degrees
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        arc_270.append((x, y))
    paths['270° Arc'] = arc_270
    expected['270° Arc'] = {
        'should_detect': True,
        'expected_arcs': 1,
        'expected_span': 270.0,
        'description': 'Single 270° arc'
    }

    # 3. S-curve (smooth continuous curve with changing curvature direction) - SHOULD DETECT 2 arcs
    # Use parametric sine wave to create smooth S-shape
    s_curve = []
    amplitude = 30
    height = 120
    for i in range(50):
        t = i / 49  # 0 to 1
        x = 100 + amplitude * math.sin(t * 2 * math.pi)
        y = 100 + t * height
        s_curve.append((x, y))
    paths['S-Curve'] = s_curve
    expected['S-Curve'] = {'should_detect': True, 'expected_arcs': 2, 'description': 'Two arcs with opposite curvature'}

    # 4. Line-Arc-Line pattern - SHOULD DETECT 1 arc (ignore straight sections)
    line_arc_line = []
    # Straight line
    for i in range(10):
        line_arc_line.append((50 + i * 3, 100))
    # Arc
    for i in range(20):
        angle = (i / 19) * math.pi
        x = 80 + 20 * math.cos(angle)
        y = 100 + 20 * math.sin(angle)
        line_arc_line.append((x, y))
    # Straight line
    for i in range(10):
        line_arc_line.append((100 + i * 3, 100))
    paths['Line-Arc-Line'] = line_arc_line
    expected['Line-Arc-Line'] = {'should_detect': True, 'expected_arcs': 1, 'description': 'One arc between straight lines'}

    return paths, expected

def visualize_curvature_segmentation(points, detector, ax):
    """Visualize the curvature segmentation step"""
    points_obj = [Point(p[0], p[1]) for p in points]

    # Apply smoothing if needed
    if detector.enable_smoothing and len(points_obj) >= detector.smoothing_window:
        if detector._detect_zigzag_pattern(points_obj):
            points_obj = detector._smooth_polyline(points_obj)

    # Get curved segments
    segments = detector._segment_by_curvature(points_obj)

    # Plot original polyline
    x_vals = [p.x for p in points_obj]
    y_vals = [p.y for p in points_obj]
    ax.plot(x_vals, y_vals, 'k.-', alpha=0.2, markersize=3, linewidth=0.5, label='Original')

    # Plot each segment with different color
    colors = plt.cm.rainbow(np.linspace(0, 1, len(segments)))
    for i, segment in enumerate(segments):
        seg_x = [p.x for p in segment]
        seg_y = [p.y for p in segment]
        ax.plot(seg_x, seg_y, 'o-', color=colors[i], markersize=5, linewidth=2,
                label=f'Segment {i+1} ({len(segment)} pts)', alpha=0.8)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(f'Curvature Segmentation\n{len(segments)} segment(s) detected')

def visualize_arc_fitting(points, detector, ax):
    """Visualize the arc fitting process"""
    points_obj = [Point(p[0], p[1]) for p in points]

    # Apply smoothing if needed
    if detector.enable_smoothing and len(points_obj) >= detector.smoothing_window:
        if detector._detect_zigzag_pattern(points_obj):
            points_obj = detector._smooth_polyline(points_obj)

    # Get segments
    segments = detector._segment_by_curvature(points_obj)

    # Plot original polyline
    x_vals = [p.x for p in points_obj]
    y_vals = [p.y for p in points_obj]
    ax.plot(x_vals, y_vals, 'k.-', alpha=0.2, markersize=2, linewidth=0.5)

    # Fit and visualize arcs
    colors = plt.cm.rainbow(np.linspace(0, 1, len(segments)))
    arc_count = 0

    for i, segment in enumerate(segments):
        if len(segment) < detector.min_arc_points:
            continue

        arc = detector._fit_arc_to_segment(segment)

        if arc:
            arc_count += 1
            # Plot the segment points
            seg_x = [p.x for p in segment]
            seg_y = [p.y for p in segment]
            ax.plot(seg_x, seg_y, 'o', color=colors[i], markersize=4, alpha=0.6)

            # Plot fitted circle/arc
            cx, cy = arc.center.x, arc.center.y
            r = arc.radius

            # Draw circle center
            ax.plot(cx, cy, 'x', color=colors[i], markersize=10, markeredgewidth=2)

            # Draw fitted arc
            theta = np.linspace(math.radians(arc.start_angle), math.radians(arc.end_angle), 100)
            arc_x = cx + r * np.cos(theta)
            arc_y = cy + r * np.sin(theta)
            ax.plot(arc_x, arc_y, '-', color=colors[i], linewidth=2, alpha=0.8,
                   label=f'Arc {arc_count}: r={r:.1f}, {arc.start_angle:.0f}°→{arc.end_angle:.0f}°')

            # Draw radius lines
            start_x = cx + r * math.cos(math.radians(arc.start_angle))
            start_y = cy + r * math.sin(math.radians(arc.start_angle))
            end_x = cx + r * math.cos(math.radians(arc.end_angle))
            end_y = cy + r * math.sin(math.radians(arc.end_angle))

            ax.plot([cx, start_x], [cy, start_y], '--', color=colors[i], alpha=0.4, linewidth=1)
            ax.plot([cx, end_x], [cy, end_y], '--', color=colors[i], alpha=0.4, linewidth=1)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_title(f'Arc Fitting Results\n{arc_count} arc(s) fitted')

def visualize_radius_deviation(points, detector, ax):
    """Visualize radius deviation for fitted arcs"""
    arcs = detector.detect_arcs(points)

    if not arcs:
        ax.text(0.5, 0.5, 'No arcs detected', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Radius Consistency')
        return

    for i, arc in enumerate(arcs):
        radii = [arc.center.distance_to(p) for p in arc.points]
        avg_radius = arc.radius

        deviations = [(r - avg_radius) / avg_radius * 100 for r in radii]

        ax.plot(range(len(deviations)), deviations, 'o-', label=f'Arc {i+1}', markersize=4)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=detector.radius_tolerance * 100, color='red', linestyle='--',
               linewidth=1, label=f'Tolerance: ±{detector.radius_tolerance*100:.1f}%')
    ax.axhline(y=-detector.radius_tolerance * 100, color='red', linestyle='--', linewidth=1)

    ax.set_xlabel('Point Index')
    ax.set_ylabel('Radius Deviation (%)')
    ax.set_title('Radius Consistency Check')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def visualize_complete_pipeline(path_name, points, detector=None, expected_result=None):
    """Create comprehensive visualization for one path"""
    if detector is None:
        detector = ArcDetector(angle_tolerance=5.0, radius_tolerance=0.02, min_arc_points=4)

    fig = plt.figure(figsize=(16, 12))

    # 1. Input polyline
    ax1 = plt.subplot(3, 3, 1)
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    ax1.plot(x_vals, y_vals, 'bo-', markersize=4, linewidth=1)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Input: {path_name}\n{len(points)} points')

    # 2. Curvature analysis
    ax2 = plt.subplot(3, 3, 2)
    points_obj = [Point(p[0], p[1]) for p in points]
    if detector.enable_smoothing and len(points_obj) >= detector.smoothing_window:
        if detector._detect_zigzag_pattern(points_obj):
            points_obj = detector._smooth_polyline(points_obj)

    # Calculate curvature
    curvatures = []
    for i in range(1, len(points_obj) - 1):
        v1 = Point(points_obj[i].x - points_obj[i-1].x, points_obj[i].y - points_obj[i-1].y)
        v2 = Point(points_obj[i+1].x - points_obj[i].x, points_obj[i+1].y - points_obj[i].y)

        len_v1 = math.sqrt(v1.x**2 + v1.y**2)
        len_v2 = math.sqrt(v2.x**2 + v2.y**2)

        if len_v1 > 1e-6 and len_v2 > 1e-6:
            v1_norm = Point(v1.x / len_v1, v1.y / len_v1)
            v2_norm = Point(v2.x / len_v2, v2.y / len_v2)

            cross = v1_norm.x * v2_norm.y - v1_norm.y * v2_norm.x
            dot = v1_norm.x * v2_norm.x + v1_norm.y * v2_norm.y
            dot = max(-1.0, min(1.0, dot))
            angle = math.degrees(math.acos(dot))

            signed_angle = angle if cross >= 0 else -angle
            curvatures.append(signed_angle)

    ax2.plot(range(len(curvatures)), curvatures, 'g-', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.fill_between(range(len(curvatures)), 0, curvatures,
                     where=[c > 0 for c in curvatures], alpha=0.3, color='red', label='Left turn')
    ax2.fill_between(range(len(curvatures)), 0, curvatures,
                     where=[c < 0 for c in curvatures], alpha=0.3, color='blue', label='Right turn')
    ax2.set_xlabel('Point Index')
    ax2.set_ylabel('Curvature (degrees)')
    ax2.set_title('Curvature Analysis')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Segmentation
    ax3 = plt.subplot(3, 3, 3)
    visualize_curvature_segmentation(points, detector, ax3)

    # 4. Arc fitting
    ax4 = plt.subplot(3, 3, 4)
    visualize_arc_fitting(points, detector, ax4)

    # 5. Radius deviation
    ax5 = plt.subplot(3, 3, 5)
    visualize_radius_deviation(points, detector, ax5)

    # 6. Final result with full information
    ax6 = plt.subplot(3, 3, 6)
    arcs = detector.detect_arcs(points)

    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    ax6.plot(x_vals, y_vals, 'k.', alpha=0.2, markersize=3)

    # Build info text with expected vs actual
    info_text = f"Detection Results:\n"
    info_text += f"Points: {len(points)}\n"
    info_text += f"Arcs detected: {len(arcs)}\n"

    # Add expected vs actual comparison
    if expected_result:
        expected_arcs = expected_result.get('expected_arcs', '?')
        matches = (len(arcs) == expected_arcs)
        status = "✓ PASS" if matches else "✗ FAIL"
        status_color = "green" if matches else "red"
        info_text += f"\n{'='*30}\n"
        info_text += f"Expected: {expected_arcs} arc(s)\n"
        info_text += f"Actual:   {len(arcs)} arc(s)\n"
        info_text += f"Status:   {status}\n"
        info_text += f"{'='*30}\n"

    info_text += "\n"

    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(arcs), 1)))

    for i, arc in enumerate(arcs):
        arc_type = detector.classify_arc(arc)
        info_text += f"Arc {i+1}: {arc_type}\n"
        info_text += f"  Center: ({arc.center.x:.1f}, {arc.center.y:.1f})\n"
        info_text += f"  Radius: {arc.radius:.2f}\n"
        info_text += f"  Angle: {arc.start_angle:.0f}°→{arc.end_angle:.0f}°\n"
        info_text += f"  Points: {len(arc.points)}\n\n"

        # Draw the arc
        cx, cy = arc.center.x, arc.center.y
        r = arc.radius
        theta = np.linspace(math.radians(arc.start_angle), math.radians(arc.end_angle), 100)
        arc_x = cx + r * np.cos(theta)
        arc_y = cy + r * np.sin(theta)
        ax6.plot(arc_x, arc_y, '-', color=colors[i], linewidth=3, alpha=0.8)
        ax6.plot(cx, cy, 'x', color=colors[i], markersize=10, markeredgewidth=2)

    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)
    ax6.set_title('Final Detection Result')

    # 7. Info panel
    ax7 = plt.subplot(3, 3, 7)
    ax7.axis('off')

    # Determine box color based on pass/fail
    box_color = 'wheat'
    if expected_result:
        expected_arcs = expected_result.get('expected_arcs', None)
        if expected_arcs is not None:
            box_color = 'lightgreen' if len(arcs) == expected_arcs else 'lightcoral'

    ax7.text(0.1, 0.9, info_text, transform=ax7.transAxes, verticalalignment='top',
             fontfamily='monospace', fontsize=9,
             bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.5))

    # 8. Algorithm parameters
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    params_text = "Algorithm Parameters:\n\n"
    params_text += f"Angle tolerance: {detector.angle_tolerance}°\n"
    params_text += f"Radius tolerance: {detector.radius_tolerance*100}%\n"
    params_text += f"Min arc points: {detector.min_arc_points}\n"
    params_text += f"Smoothing: {detector.enable_smoothing}\n"
    params_text += f"Smoothing window: {detector.smoothing_window}\n\n"
    params_text += "Detection Method:\n"
    params_text += "• Global circle detection\n"
    params_text += "• AASR (Angle-based\n"
    params_text += "  Segmentation &\n"
    params_text += "  Reconstruction)\n"

    ax8.text(0.1, 0.9, params_text, transform=ax8.transAxes, verticalalignment='top',
             fontfamily='monospace', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # 9. Comparison: Original vs Reconstructed
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(x_vals, y_vals, 'k.-', alpha=0.3, markersize=3, linewidth=0.5, label='Original polyline')

    for i, arc in enumerate(arcs):
        cx, cy = arc.center.x, arc.center.y
        r = arc.radius
        theta = np.linspace(math.radians(arc.start_angle), math.radians(arc.end_angle), 100)
        arc_x = cx + r * np.cos(theta)
        arc_y = cy + r * np.sin(theta)
        ax9.plot(arc_x, arc_y, '-', color=colors[i], linewidth=3, alpha=0.8, label=f'Arc {i+1} (fitted)')

    ax9.set_aspect('equal')
    ax9.grid(True, alpha=0.3)
    ax9.legend(fontsize=8)
    ax9.set_title('Original vs Reconstructed')

    # Add expected result description to title
    title = f'AASR Arc Detection Pipeline: {path_name}'
    if expected_result:
        desc = expected_result.get('description', '')
        expected_arcs = expected_result.get('expected_arcs', '?')
        matches = (len(arcs) == expected_arcs)
        status = "✓ PASS" if matches else "✗ FAIL"
        title += f'\nExpected: {desc} | Result: {status}'

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def print_test_header():
    """Print standardized test header"""
    print("\n" + "="*80)
    print("ARC DETECTION TEST SUITE - AASR Algorithm")
    print("="*80)

def print_test_result(test_name, expected, actual, details=None):
    """Print standardized test result"""
    print("\n" + "-"*80)
    print(f"TEST: {test_name}")
    if details:
        print(f"DESCRIPTION: {details.get('description', 'N/A')}")
    print("-"*80)
    print(f"EXPECTED: {expected}")
    print(f"ACTUAL:   {actual}")

    matches = (expected == actual)
    status = "✓ PASS" if matches else "✗ FAIL"
    print(f"STATUS:   {status}")

    if details and 'additional_info' in details:
        print(f"DETAILS:  {details['additional_info']}")
    print("-"*80)

    return matches


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize AASR arc detection algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python visualize_arc_detection.py
  python visualize_arc_detection.py --angle-tolerance 10.0
  python visualize_arc_detection.py --radius-tolerance 0.05 --min-arc-points 6
        '''
    )

    parser.add_argument('--angle-tolerance', type=float, default=5.0, help='Angle tolerance in degrees (default: 5.0)')
    parser.add_argument('--radius-tolerance', type=float, default=0.02, help='Radius tolerance fraction (default: 0.02)')
    parser.add_argument('--min-arc-points', type=int, default=4, help='Minimum arc points (default: 4)')
    parser.add_argument('--smoothing-window', type=int, default=5, help='Smoothing window size (default: 5)')
    parser.add_argument('--no-smoothing', action='store_true', help='Disable smoothing')
    parser.add_argument('--dpi', type=int, default=150, help='Output DPI (default: 150)')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots (just save to files)')

    args = parser.parse_args()

    # Create detector with specified parameters
    detector = ArcDetector(
        angle_tolerance=args.angle_tolerance,
        radius_tolerance=args.radius_tolerance,
        min_arc_points=args.min_arc_points,
        enable_smoothing=not args.no_smoothing,
        smoothing_window=args.smoothing_window
    )

    paths, expected_results = create_test_paths()

    # Print test header
    print_test_header()
    print(f"Algorithm: AASR (Angle-based Arc Segmentation & Reconstruction)")
    print(f"Parameters:")
    print(f"  - Angle tolerance: {args.angle_tolerance}°")
    print(f"  - Radius tolerance: {args.radius_tolerance*100}%")
    print(f"  - Min arc points: {args.min_arc_points}")
    print(f"  - Smoothing enabled: {not args.no_smoothing}")

    # Track test results
    test_results = []

    # Store comprehensive results for summary
    comprehensive_results = []

    for path_name, points in paths.items():
        expected = expected_results.get(path_name)

        # Run detection
        arcs = detector.detect_arcs(points)

        # Enhanced validation with comprehensive metrics
        expected_arcs = expected.get('expected_arcs', 0)
        expected_span = expected.get('expected_span', None)

        analysis = analyze_detection_quality(
            points,
            arcs,
            expected_arcs,
            expected_span
        )

        # Print enhanced test result
        print(format_test_result(path_name, analysis, expected_arcs, arcs))

        # Store for summary
        test_results.append(analysis['overall_pass'])
        comprehensive_results.append((path_name, analysis))

        # Generate visualization
        print(f"Generating visualization...")
        fig = visualize_complete_pipeline(path_name, points, detector, expected)

        # Save
        safe_name = path_name.replace('°', 'deg').replace(' ', '_').replace('-', '_')
        filename = f'output/arc_detection_{safe_name}.png'
        plt.savefig(filename, dpi=args.dpi, bbox_inches='tight')
        print(f"✓ Saved: {filename}")

    # Print enhanced summary
    print_summary(comprehensive_results)

    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()
