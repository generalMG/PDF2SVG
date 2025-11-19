#!/usr/bin/env python3
"""
Complete Pipeline Visualization

Interactive demonstration of the entire PDF2SVG arc detection pipeline:
1. Load test polylines (or create synthetic ones)
2. Show preprocessing (zigzag detection & smoothing)
3. Show hybrid detection (Global → AASR fallback)
4. Show arc merging
5. Compare input vs output

This is the master visualization showing how everything works together.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
import math
from arc_detector import ArcDetector, Point

def create_comprehensive_test_case():
    """Create a complex test case with multiple geometric elements"""
    # Create a drawing with:
    # 1. A noisy full circle
    # 2. A clean 180-degree arc
    # 3. An S-curve
    # 4. A straight line

    test_cases = {}
    expected = {}

    center_x, center_y = 150, 150
    radius = 40

    # 1. Noisy full circle - SHOULD DETECT 1 circle via global detection
    noisy_circle = []
    for i in range(60):
        angle = (i / 60) * 2 * math.pi
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)

        # Add zigzag noise
        noise = 2.0 * (1 if i % 2 == 0 else -1)
        x += noise * math.cos(angle + math.pi/2)
        y += noise * math.sin(angle + math.pi/2)

        noisy_circle.append((x, y))

    test_cases['Noisy Circle'] = noisy_circle
    expected['Noisy Circle'] = {
        'expected_arcs': 1,
        'description': 'Full 360° CIRCLE with noise',
        'test_type': 'CIRCLE',
        'should_use_global': True,
        'explanation': 'Should be detected as closed loop by Global detection, returned as 1 circle'
    }

    # 2. Clean 180-degree arc (semicircle) - SHOULD DETECT 1 arc
    clean_arc = []
    for i in range(30):
        angle = (i / 29) * math.pi
        x = 300 + 35 * math.cos(angle)
        y = 150 + 35 * math.sin(angle)
        clean_arc.append((x, y))

    test_cases['Clean Semicircle'] = clean_arc
    expected['Clean Semicircle'] = {
        'expected_arcs': 1,
        'description': 'Single 180° arc',
        'test_type': 'ARC',
        'should_use_global': False,
        'explanation': 'Partial arc, should be detected by AASR (not a closed loop)'
    }

    # 3. S-curve (smooth continuous curve with changing curvature direction) - SHOULD DETECT 2 arcs
    # Use parametric sine wave to create smooth S-shape
    s_curve = []
    amplitude = 25
    height = 100
    base_y = 280
    for i in range(40):
        t = i / 39  # 0 to 1
        x = 100 + amplitude * math.sin(t * 2 * math.pi)
        y = base_y + t * height
        s_curve.append((x, y))

    test_cases['S-Curve'] = s_curve
    expected['S-Curve'] = {
        'expected_arcs': 2,
        'description': 'Two arcs with opposite curvature',
        'test_type': 'COMPOSITE',
        'should_use_global': False,
        'explanation': 'S-curve should be segmented into 2 arcs by AASR'
    }

    # 4. Straight line - SHOULD DETECT 0 arcs
    straight_line = []
    for i in range(20):
        straight_line.append((250 + i * 3, 280))

    test_cases['Straight Line'] = straight_line
    expected['Straight Line'] = {
        'expected_arcs': 0,
        'description': 'Straight line - no arcs',
        'test_type': 'LINE',
        'should_use_global': False,
        'explanation': 'Straight line should not be detected as arc by either method'
    }

    return test_cases, expected

def visualize_pipeline_step_by_step(points, title, detector, expected_result=None):
    """Create a comprehensive step-by-step visualization"""
    fig = plt.figure(figsize=(20, 14))

    # Convert to Point objects
    points_obj = [Point(p[0], p[1]) for p in points]

    # Step 1: Original input
    ax1 = plt.subplot(3, 4, 1)
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    ax1.plot(x_vals, y_vals, 'bo-', markersize=4, linewidth=1)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Step 1: Input\n{len(points)} points')

    # Step 2: Zigzag detection
    ax2 = plt.subplot(3, 4, 2)
    is_zigzag = detector._detect_zigzag_pattern(points_obj)

    # Calculate angle changes for visualization
    angle_changes = []
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
            angle_changes.append(signed_angle)

    colors = ['red' if a > 0 else 'blue' for a in angle_changes]
    ax2.bar(range(len(angle_changes)), angle_changes, color=colors, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Point Index')
    ax2.set_ylabel('Angle (°)')
    ax2.set_title(f'Step 2: Zigzag Detection\n{"Zigzag detected!" if is_zigzag else "No zigzag"}')
    ax2.grid(True, alpha=0.3)

    # Step 3: Smoothing (if needed)
    ax3 = plt.subplot(3, 4, 3)
    if is_zigzag and detector.enable_smoothing:
        smoothed = detector._smooth_polyline(points_obj)
        x_smooth = [p.x for p in smoothed]
        y_smooth = [p.y for p in smoothed]
        ax3.plot(x_vals, y_vals, 'r.-', alpha=0.3, markersize=3, linewidth=0.5, label='Original')
        ax3.plot(x_smooth, y_smooth, 'go-', markersize=4, linewidth=1, label='Smoothed')
        ax3.legend()
        points_for_detection = smoothed
        smoothing_text = "Applied (3 passes)"
    else:
        ax3.plot(x_vals, y_vals, 'bo-', markersize=4, linewidth=1)
        points_for_detection = points_obj
        smoothing_text = "Not needed"

    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_title(f'Step 3: Smoothing\n{smoothing_text}')

    # Step 4: Global circle check
    ax4 = plt.subplot(3, 4, 4)
    global_circle = detector.detect_circle_global(points)

    if global_circle:
        cx, cy = global_circle.center.x, global_circle.center.y
        r = global_circle.radius
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = cx + r * np.cos(theta)
        circle_y = cy + r * np.sin(theta)
        ax4.plot(x_vals, y_vals, 'b.', markersize=3, alpha=0.4)
        ax4.plot(circle_x, circle_y, 'r-', linewidth=3, label=f'Detected circle\nr={r:.1f}')
        ax4.plot(cx, cy, 'r*', markersize=15)
        detection_text = "✓ Circle found"
        detection_color = 'lightgreen'
    else:
        ax4.plot(x_vals, y_vals, 'bo-', markersize=4, linewidth=1)
        detection_text = "✗ Not a circle"
        detection_color = 'lightcoral'

    ax4.text(0.5, 0.95, detection_text, transform=ax4.transAxes,
             ha='center', va='top', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor=detection_color, alpha=0.8))
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Step 4: Global Detection\n(Fast path)')
    if global_circle:
        ax4.legend(fontsize=8)

    # Step 5: AASR segmentation (if global failed)
    ax5 = plt.subplot(3, 4, 5)
    if not global_circle:
        segments = detector._segment_by_curvature(points_for_detection)
        colors = plt.cm.rainbow(np.linspace(0, 1, max(len(segments), 1)))
        for i, segment in enumerate(segments):
            seg_x = [p.x for p in segment]
            seg_y = [p.y for p in segment]
            ax5.plot(seg_x, seg_y, 'o-', color=colors[i], markersize=4, linewidth=2,
                    label=f'Seg {i+1} ({len(segment)}pts)')
        ax5.set_title(f'Step 5: AASR Segmentation\n{len(segments)} segment(s)')
        ax5.legend(fontsize=7)
    else:
        ax5.text(0.5, 0.5, 'Skipped\n(Global detection succeeded)', ha='center', va='center',
                transform=ax5.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax5.set_title('Step 5: AASR Segmentation\n(Skipped)')

    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)

    # Step 6: Arc fitting
    ax6 = plt.subplot(3, 4, 6)
    if global_circle:
        arcs = [global_circle]
    else:
        arcs = detector.detect_arcs(points)

    if arcs:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(arcs)))
        for i, arc in enumerate(arcs):
            cx, cy = arc.center.x, arc.center.y
            r = arc.radius
            theta = np.linspace(math.radians(arc.start_angle), math.radians(arc.end_angle), 100)
            arc_x = cx + r * np.cos(theta)
            arc_y = cy + r * np.sin(theta)
            ax6.plot(arc_x, arc_y, '-', color=colors[i], linewidth=3, label=f'Arc {i+1}')
            ax6.plot(cx, cy, 'x', color=colors[i], markersize=12, markeredgewidth=2)

        ax6.plot(x_vals, y_vals, 'k.', markersize=2, alpha=0.3)
        ax6.legend(fontsize=7)
    else:
        ax6.plot(x_vals, y_vals, 'bo-', markersize=4, linewidth=1)

    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)
    ax6.set_title(f'Step 6: Arc Fitting\n{len(arcs)} arc(s) detected')

    # Step 7: Arc classification
    ax7 = plt.subplot(3, 4, 7)
    ax7.axis('off')

    if arcs:
        info_text = "Arc Details:\n\n"
        for i, arc in enumerate(arcs):
            arc_type = detector.classify_arc(arc)
            angle_span = (arc.end_angle - arc.start_angle) % 360
            info_text += f"Arc {i+1}:\n"
            info_text += f"  Type: {arc_type}\n"
            info_text += f"  Center: ({arc.center.x:.1f}, {arc.center.y:.1f})\n"
            info_text += f"  Radius: {arc.radius:.2f}\n"
            info_text += f"  Span: {angle_span:.0f}°\n"
            info_text += f"  Points: {len(arc.points)}\n\n"
    else:
        info_text = "No arcs detected.\n\nPossible reasons:\n"
        info_text += "• Straight line\n"
        info_text += "• Insufficient points\n"
        info_text += "• Irregular curvature\n"

    # Add expected vs actual comparison
    if expected_result:
        expected_arcs = expected_result.get('expected_arcs', '?')
        matches = (len(arcs) == expected_arcs)
        status = "✓ PASS" if matches else "✗ FAIL"
        info_text += f"\n{'='*30}\n"
        info_text += f"Expected: {expected_arcs} arc(s)\n"
        info_text += f"Actual:   {len(arcs)} arc(s)\n"
        info_text += f"Status:   {status}\n"

    # Determine box color based on pass/fail
    box_color = 'wheat'
    if expected_result:
        expected_arcs = expected_result.get('expected_arcs', None)
        if expected_arcs is not None:
            box_color = 'lightgreen' if len(arcs) == expected_arcs else 'lightcoral'

    ax7.text(0.1, 0.9, info_text, transform=ax7.transAxes, verticalalignment='top',
             fontfamily='monospace', fontsize=9,
             bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.5))
    ax7.set_title('Step 7: Classification')

    # Step 8: Comparison (Original vs Detected)
    ax8 = plt.subplot(3, 4, 8)
    ax8.plot(x_vals, y_vals, 'k.-', alpha=0.3, markersize=2, linewidth=0.5, label='Original')

    if arcs:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(arcs)))
        for i, arc in enumerate(arcs):
            cx, cy = arc.center.x, arc.center.y
            r = arc.radius
            theta = np.linspace(math.radians(arc.start_angle), math.radians(arc.end_angle), 100)
            arc_x = cx + r * np.cos(theta)
            arc_y = cy + r * np.sin(theta)
            ax8.plot(arc_x, arc_y, '-', color=colors[i], linewidth=4, alpha=0.8)

    ax8.set_aspect('equal')
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=8)
    ax8.set_title('Step 8: Result\nOriginal vs Reconstructed')

    # Row 3: Additional analysis

    # Step 9: Radius consistency
    ax9 = plt.subplot(3, 4, 9)
    if arcs:
        for i, arc in enumerate(arcs):
            radii = [arc.center.distance_to(p) for p in arc.points]
            deviations = [(r - arc.radius) / arc.radius * 100 for r in radii]
            ax9.plot(range(len(deviations)), deviations, 'o-', markersize=3, label=f'Arc {i+1}')

        ax9.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax9.axhline(y=detector.radius_tolerance * 100, color='red', linestyle='--', linewidth=1)
        ax9.axhline(y=-detector.radius_tolerance * 100, color='red', linestyle='--', linewidth=1)
        ax9.set_xlabel('Point Index')
        ax9.set_ylabel('Deviation (%)')
        ax9.legend(fontsize=7)
    else:
        ax9.text(0.5, 0.5, 'No arcs to analyze', ha='center', va='center', transform=ax9.transAxes)

    ax9.grid(True, alpha=0.3)
    ax9.set_title('Radius Consistency')

    # Step 10: Algorithm parameters
    ax10 = plt.subplot(3, 4, 10)
    ax10.axis('off')

    params_text = "Algorithm Parameters:\n\n"
    params_text += f"Angle tolerance: {detector.angle_tolerance}°\n"
    params_text += f"Radius tolerance: {detector.radius_tolerance*100}%\n"
    params_text += f"Min arc points: {detector.min_arc_points}\n"
    params_text += f"Smoothing enabled: {detector.enable_smoothing}\n"
    params_text += f"Smoothing window: {detector.smoothing_window}\n\n"
    params_text += "Detection Pipeline:\n"
    params_text += "1. Zigzag detection\n"
    params_text += "2. Smoothing (if needed)\n"
    params_text += "3. Global circle check\n"
    params_text += "4. AASR fallback\n"
    params_text += "5. Arc classification\n"

    ax10.text(0.1, 0.9, params_text, transform=ax10.transAxes, verticalalignment='top',
              fontfamily='monospace', fontsize=8,
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax10.set_title('Algorithm Configuration')

    # Step 11: Statistics
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')

    stats_text = "Conversion Statistics:\n\n"
    stats_text += f"Input:\n"
    stats_text += f"  Points: {len(points)}\n"
    stats_text += f"  Line segments: {len(points) - 1}\n\n"
    stats_text += f"Output:\n"
    stats_text += f"  Arcs: {len(arcs)}\n"

    if arcs:
        total_points_in_arcs = sum(len(arc.points) for arc in arcs)
        total_segments_saved = sum(len(arc.points) - 1 for arc in arcs)
        reduction = (1 - len(arcs) / max(1, total_segments_saved)) * 100

        stats_text += f"  Points in arcs: {total_points_in_arcs}\n"
        stats_text += f"  Segments saved: {total_segments_saved}\n"
        stats_text += f"  Reduction: {reduction:.1f}%\n\n"
        stats_text += f"File size impact:\n"
        stats_text += f"  SVG elements: {len(points)-1} → {len(arcs)}\n"

    ax11.text(0.1, 0.9, stats_text, transform=ax11.transAxes, verticalalignment='top',
              fontfamily='monospace', fontsize=9,
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax11.set_title('Statistics')

    # Step 12: Decision flow
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')

    flow_text = "Detection Flow:\n\n"
    flow_text += f"1. Input: {len(points)} points\n"
    flow_text += f"   ↓\n"
    flow_text += f"2. Zigzag? {'Yes' if is_zigzag else 'No'}\n"
    flow_text += f"   ↓\n"
    if is_zigzag and detector.enable_smoothing:
        flow_text += f"3. Smoothing: Applied\n"
    else:
        flow_text += f"3. Smoothing: Skipped\n"
    flow_text += f"   ↓\n"
    if global_circle:
        flow_text += f"4. Global: ✓ Circle found\n"
        flow_text += f"   ↓\n"
        flow_text += f"5. AASR: Skipped\n"
    else:
        flow_text += f"4. Global: ✗ Not a circle\n"
        flow_text += f"   ↓\n"
        flow_text += f"5. AASR: {len(arcs)} arc(s)\n"
    flow_text += f"   ↓\n"
    flow_text += f"6. Output: {len(arcs)} arc(s)\n"

    ax12.text(0.1, 0.9, flow_text, transform=ax12.transAxes, verticalalignment='top',
              fontfamily='monospace', fontsize=9,
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax12.set_title('Decision Flow')

    # Add expected result to title
    test_type = expected_result.get('test_type', 'UNKNOWN') if expected_result else 'UNKNOWN'
    main_title = f'Pipeline Test [{test_type}]: {title}'
    if expected_result:
        desc = expected_result.get('description', '')
        expected_arcs = expected_result.get('expected_arcs', '?')
        should_use_global = expected_result.get('should_use_global', False)
        matches = (len(arcs) == expected_arcs)
        status = "✓ PASS" if matches else "✗ FAIL"
        method_info = f"Method: {'Global' if global_circle else 'AASR'} (Expected: {'Global' if should_use_global else 'AASR'})"
        main_title += f'\n{desc} | {method_info} | Result: {status}'

    plt.suptitle(main_title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    return fig

def print_test_header():
    """Print standardized test header"""
    print("\n" + "="*80)
    print("COMPLETE PIPELINE TEST SUITE - Hybrid Detection")
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

def print_summary(total, passed, failed):
    """Print standardized test summary"""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests:  {total}")
    print(f"Passed:       {passed} ({(passed/total*100):.1f}%)")
    print(f"Failed:       {failed} ({(failed/total*100):.1f}%)")
    print("="*80 + "\n")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize complete arc detection pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python visualize_complete_pipeline.py
  python visualize_complete_pipeline.py --angle-tolerance 10.0
  python visualize_complete_pipeline.py --radius-tolerance 0.05 --no-smoothing
        '''
    )

    parser.add_argument('--angle-tolerance', type=float, default=5.0, help='Angle tolerance in degrees (default: 5.0)')
    parser.add_argument('--radius-tolerance', type=float, default=0.02, help='Radius tolerance fraction (default: 0.02)')
    parser.add_argument('--min-arc-points', type=int, default=4, help='Minimum arc points (default: 4)')
    parser.add_argument('--smoothing-window', type=int, default=5, help='Smoothing window size (default: 5)')
    parser.add_argument('--no-smoothing', action='store_true', help='Disable smoothing')
    parser.add_argument('--dpi', type=int, default=150, help='Output DPI (default: 150)')

    args = parser.parse_args()

    # Print test header
    print_test_header()
    print(f"Algorithm: Hybrid (Global Circle Detection → AASR Fallback)")
    print(f"Parameters:")
    print(f"  - Angle tolerance: {args.angle_tolerance}°")
    print(f"  - Radius tolerance: {args.radius_tolerance*100}%")
    print(f"  - Min arc points: {args.min_arc_points}")
    print(f"  - Smoothing enabled: {not args.no_smoothing}")
    print(f"  - Smoothing window: {args.smoothing_window}")

    detector = ArcDetector(
        angle_tolerance=args.angle_tolerance,
        radius_tolerance=args.radius_tolerance,
        min_arc_points=args.min_arc_points,
        enable_smoothing=not args.no_smoothing,
        smoothing_window=args.smoothing_window
    )

    test_cases, expected_results = create_comprehensive_test_case()

    # Track test results
    test_results = []

    for name, points in test_cases.items():
        expected = expected_results.get(name)

        # Run detection
        arcs = detector.detect_arcs(points)
        global_result = detector.detect_circle_global(points)

        # Print test result
        expected_arcs = expected.get('expected_arcs', 0)
        actual_arcs = len(arcs)
        used_global = (global_result is not None)
        should_use_global = expected.get('should_use_global', False)
        test_type = expected.get('test_type', 'UNKNOWN')

        # Build detailed description
        details = {
            'description': expected.get('description', 'N/A'),
            'additional_info': (
                f"Test Type: {test_type} | "
                f"Expected Method: {'Global (circle detection)' if should_use_global else 'AASR (arc segmentation)'} | "
                f"Actual Method Used: {'Global' if used_global else 'AASR'} | "
                f"Result: {actual_arcs} arc(s) detected from {len(points)} points"
            )
        }

        # Add explanation
        if 'explanation' in expected:
            details['additional_info'] += f"\nExpected Behavior: {expected['explanation']}"

        passed = print_test_result(
            name,
            f"{expected_arcs} arc(s) [{test_type}]",
            f"{actual_arcs} arc(s) [via {'Global' if used_global else 'AASR'}]",
            details
        )

        test_results.append(passed)

        # Generate visualization
        print(f"Generating visualization...")
        fig = visualize_pipeline_step_by_step(points, name, detector, expected)

        # Save
        safe_name = name.replace(' ', '_').replace('-', '_')
        filename = f'output/pipeline_{safe_name}.png'
        plt.savefig(filename, dpi=args.dpi, bbox_inches='tight')
        print(f"✓ Saved: {filename}")

    # Print summary
    total = len(test_results)
    passed = sum(test_results)
    failed = total - passed
    print_summary(total, passed, failed)

    plt.show()

if __name__ == "__main__":
    main()
