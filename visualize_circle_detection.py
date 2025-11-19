#!/usr/bin/env python3
"""
Visualization of Global Circle Detection Algorithm

Shows how the global circle detection works:
1. Closed loop check (first point ≈ last point)
2. Centroid calculation
3. Radius consistency check
4. Comparison: Global vs AASR detection
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from arc_detector import ArcDetector, Point

def create_circles_with_varying_resolution():
    """Create circles with different point counts"""
    circles = {}
    center_x, center_y = 100, 100
    radius = 50

    for num_points in [12, 24, 50, 100, 200]:
        points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y))

        circles[f'{num_points} points'] = points

    return circles

def create_partial_arcs():
    """Create partial arcs (not closed loops)"""
    arcs = {}
    center_x, center_y = 100, 100
    radius = 50

    # 90 degree arc
    arc_90 = []
    for i in range(25):
        angle = (i / 24) * (math.pi / 2)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        arc_90.append((x, y))
    arcs['90° arc'] = arc_90

    # 270 degree arc
    arc_270 = []
    for i in range(75):
        angle = (i / 74) * (3 * math.pi / 2)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        arc_270.append((x, y))
    arcs['270° arc'] = arc_270

    # Almost complete (350 degrees)
    arc_350 = []
    for i in range(95):
        angle = (i / 94) * (350 * math.pi / 180)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        arc_350.append((x, y))
    arcs['350° arc'] = arc_350

    return arcs

def visualize_closed_loop_check(points, detector, ax):
    """Visualize closed loop check"""
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]

    # Plot polyline
    ax.plot(x_vals, y_vals, 'b.-', markersize=4, linewidth=1, alpha=0.6)

    # Highlight first and last points
    ax.plot(x_vals[0], y_vals[0], 'go', markersize=12, label='First point', zorder=5)
    ax.plot(x_vals[-1], y_vals[-1], 'ro', markersize=12, label='Last point', zorder=5)

    # Draw connection line
    ax.plot([x_vals[0], x_vals[-1]], [y_vals[0], y_vals[-1]], 'r--',
            linewidth=2, alpha=0.5, label='Closure gap')

    # Calculate closure distance
    dx = points[-1][0] - points[0][0]
    dy = points[-1][1] - points[0][1]
    closure_dist = math.sqrt(dx**2 + dy**2)

    # Calculate average segment length
    total_length = sum(math.sqrt((points[i+1][0] - points[i][0])**2 +
                                 (points[i+1][1] - points[i][1])**2)
                      for i in range(len(points) - 1))
    avg_segment_length = total_length / (len(points) - 1)

    # Check if closed
    is_closed = detector.is_closed_loop(points)

    # Info text
    info_text = f"Points: {len(points)}\n"
    info_text += f"Closure distance: {closure_dist:.2f}\n"
    info_text += f"Avg segment length: {avg_segment_length:.2f}\n"
    info_text += f"Tolerance: 1.5 × avg = {1.5 * avg_segment_length:.2f}\n"
    info_text += f"Is closed: {is_closed}"

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontfamily='monospace', fontsize=9)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title('Closed Loop Check')

def visualize_centroid_calculation(points, detector, ax):
    """Visualize centroid calculation"""
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]

    # Plot polyline
    ax.plot(x_vals, y_vals, 'b.-', markersize=3, linewidth=1, alpha=0.6)

    # Calculate centroid
    points_obj = [Point(p[0], p[1]) for p in points]
    cx = sum(p.x for p in points_obj) / len(points_obj)
    cy = sum(p.y for p in points_obj) / len(points_obj)

    # Plot centroid
    ax.plot(cx, cy, 'r*', markersize=20, label='Centroid', zorder=5)

    # Draw radii to some points
    step = max(1, len(points) // 8)
    for i in range(0, len(points), step):
        ax.plot([cx, x_vals[i]], [cy, y_vals[i]], 'g--', alpha=0.3, linewidth=0.5)

    # Calculate radii
    radii = [math.sqrt((p[0] - cx)**2 + (p[1] - cy)**2) for p in points]
    avg_radius = sum(radii) / len(radii)

    # Draw average radius circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = cx + avg_radius * np.cos(theta)
    circle_y = cy + avg_radius * np.sin(theta)
    ax.plot(circle_x, circle_y, 'r--', linewidth=2, alpha=0.5, label=f'Avg radius: {avg_radius:.2f}')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title('Centroid Calculation')

def visualize_radius_consistency_check(points, detector, ax):
    """Visualize radius consistency check"""
    result = detector.check_radius_consistency(points)

    if result is None:
        ax.text(0.5, 0.5, 'No result', ha='center', va='center', transform=ax.transAxes)
        return

    center, avg_radius, rel_deviation = result
    points_obj = [Point(p[0], p[1]) for p in points]

    # Calculate radii
    radii = [center.distance_to(p) for p in points_obj]

    # Plot radius values
    ax.plot(range(len(radii)), radii, 'b-', linewidth=1, label='Radius to each point')
    ax.axhline(y=avg_radius, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_radius:.2f}')

    # Show tolerance bands
    tolerance = detector.radius_tolerance
    ax.axhline(y=avg_radius * (1 + tolerance), color='orange', linestyle=':',
               linewidth=1, label=f'±{tolerance*100}% tolerance')
    ax.axhline(y=avg_radius * (1 - tolerance), color='orange', linestyle=':', linewidth=1)

    # Highlight points outside tolerance
    outside_tolerance = []
    for i, r in enumerate(radii):
        if abs(r - avg_radius) / avg_radius > tolerance:
            outside_tolerance.append(i)

    if outside_tolerance:
        ax.plot(outside_tolerance, [radii[i] for i in outside_tolerance],
                'rx', markersize=10, label='Outside tolerance')

    # Fill tolerance band
    ax.fill_between(range(len(radii)),
                     avg_radius * (1 - tolerance),
                     avg_radius * (1 + tolerance),
                     alpha=0.2, color='green')

    # Info text
    info_text = f"Average radius: {avg_radius:.2f}\n"
    info_text += f"Relative deviation: {rel_deviation*100:.2f}%\n"
    info_text += f"Tolerance: {tolerance*100}%\n"
    info_text += f"Points outside: {len(outside_tolerance)}/{len(radii)}\n"
    info_text += f"Passes check: {rel_deviation <= tolerance}"

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontfamily='monospace', fontsize=9)

    ax.set_xlabel('Point Index')
    ax.set_ylabel('Radius')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title('Radius Consistency Check')

def visualize_global_vs_aasr(points, detector, ax1, ax2):
    """Compare global and AASR detection"""
    # Global detection
    global_result = detector.detect_circle_global(points)

    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]

    # Plot on ax1 (Global)
    ax1.plot(x_vals, y_vals, 'b.', markersize=4, alpha=0.4, label='Input points')

    if global_result:
        cx, cy = global_result.center.x, global_result.center.y
        r = global_result.radius

        # Draw detected circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = cx + r * np.cos(theta)
        circle_y = cy + r * np.sin(theta)
        ax1.plot(circle_x, circle_y, 'r-', linewidth=3, alpha=0.8, label=f'Detected: r={r:.2f}')
        ax1.plot(cx, cy, 'r*', markersize=15, label='Center')

        result_text = "✓ Circle detected"
    else:
        result_text = "✗ No circle detected"

    ax1.text(0.5, 0.95, result_text, transform=ax1.transAxes,
             ha='center', va='top', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen' if global_result else 'lightcoral', alpha=0.8))

    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_title('Global Circle Detection\n(Fast O(n))')

    # AASR detection
    aasr_arcs = detector.detect_arcs(points)

    # Plot on ax2 (AASR)
    ax2.plot(x_vals, y_vals, 'b.', markersize=4, alpha=0.4, label='Input points')

    if aasr_arcs:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(aasr_arcs)))
        for i, arc in enumerate(aasr_arcs):
            cx, cy = arc.center.x, arc.center.y
            r = arc.radius
            theta = np.linspace(math.radians(arc.start_angle), math.radians(arc.end_angle), 100)
            arc_x = cx + r * np.cos(theta)
            arc_y = cy + r * np.sin(theta)
            ax2.plot(arc_x, arc_y, '-', color=colors[i], linewidth=3, alpha=0.8,
                    label=f'Arc {i+1}: r={r:.2f}')
            ax2.plot(cx, cy, 'x', color=colors[i], markersize=12, markeredgewidth=2)

        result_text = f"✓ {len(aasr_arcs)} arc(s) detected"
    else:
        result_text = "✗ No arcs detected"

    ax2.text(0.5, 0.95, result_text, transform=ax2.transAxes,
             ha='center', va='top', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen' if aasr_arcs else 'lightcoral', alpha=0.8))

    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_title('AASR Detection\n(Segmentation-based)')

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize global circle detection algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python visualize_circle_detection.py
  python visualize_circle_detection.py --angle-tolerance 10.0
  python visualize_circle_detection.py --radius-tolerance 0.05
        '''
    )

    parser.add_argument('--angle-tolerance', type=float, default=5.0, help='Angle tolerance in degrees (default: 5.0)')
    parser.add_argument('--radius-tolerance', type=float, default=0.02, help='Radius tolerance fraction (default: 0.02)')
    parser.add_argument('--min-arc-points', type=int, default=4, help='Minimum arc points (default: 4)')
    parser.add_argument('--smoothing-window', type=int, default=5, help='Smoothing window size (default: 5)')
    parser.add_argument('--no-smoothing', action='store_true', help='Disable smoothing')
    parser.add_argument('--dpi', type=int, default=150, help='Output DPI (default: 150)')

    args = parser.parse_args()

    detector = ArcDetector(
        angle_tolerance=args.angle_tolerance,
        radius_tolerance=args.radius_tolerance,
        min_arc_points=args.min_arc_points,
        enable_smoothing=not args.no_smoothing,
        smoothing_window=args.smoothing_window
    )

    # Test 1: Circles with varying resolution
    print("\n" + "="*60)
    print("Test 1: Circles with varying resolution")
    print("="*60)

    circles = create_circles_with_varying_resolution()

    for circle_name, points in circles.items():
        print(f"\nProcessing: {circle_name}")

        fig = plt.figure(figsize=(16, 10))

        # Row 1: Input and checks
        ax1 = plt.subplot(2, 3, 1)
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        ax1.plot(x_vals, y_vals, 'bo-', markersize=4, linewidth=1)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Input: {circle_name}')

        ax2 = plt.subplot(2, 3, 2)
        visualize_closed_loop_check(points, detector, ax2)

        ax3 = plt.subplot(2, 3, 3)
        visualize_centroid_calculation(points, detector, ax3)

        ax4 = plt.subplot(2, 3, 4)
        visualize_radius_consistency_check(points, detector, ax4)

        # Row 2: Comparison
        ax5 = plt.subplot(2, 3, 5)
        ax6 = plt.subplot(2, 3, 6)
        visualize_global_vs_aasr(points, detector, ax5, ax6)

        plt.suptitle(f'Global Circle Detection: {circle_name}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Save
        safe_name = circle_name.replace(' ', '_')
        filename = f'output/circle_detection_{safe_name}.png'
        plt.savefig(filename, dpi=args.dpi, bbox_inches='tight')
        print(f"✓ Saved: {filename}")

    # Test 2: Partial arcs (should NOT be detected as circles)
    print("\n" + "="*60)
    print("Test 2: Partial arcs (should fail closed loop check)")
    print("="*60)

    arcs = create_partial_arcs()

    for arc_name, points in arcs.items():
        print(f"\nProcessing: {arc_name}")

        fig = plt.figure(figsize=(16, 10))

        ax1 = plt.subplot(2, 3, 1)
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        ax1.plot(x_vals, y_vals, 'bo-', markersize=4, linewidth=1)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Input: {arc_name}')

        ax2 = plt.subplot(2, 3, 2)
        visualize_closed_loop_check(points, detector, ax2)

        ax3 = plt.subplot(2, 3, 3)
        visualize_centroid_calculation(points, detector, ax3)

        ax4 = plt.subplot(2, 3, 4)
        visualize_radius_consistency_check(points, detector, ax4)

        ax5 = plt.subplot(2, 3, 5)
        ax6 = plt.subplot(2, 3, 6)
        visualize_global_vs_aasr(points, detector, ax5, ax6)

        plt.suptitle(f'Global Circle Detection: {arc_name}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Save
        safe_name = arc_name.replace('°', 'deg').replace(' ', '_')
        filename = f'output/circle_detection_{safe_name}.png'
        plt.savefig(filename, dpi=args.dpi, bbox_inches='tight')
        print(f"✓ Saved: {filename}")

    plt.show()

if __name__ == "__main__":
    main()
