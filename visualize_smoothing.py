#!/usr/bin/env python3
"""
Visualization of Zigzag Detection and Smoothing Algorithm

Shows step-by-step how the smoothing algorithm works:
1. Original noisy polyline
2. Zigzag pattern detection
3. Smoothing process (before/after each pass)
4. Final smoothed result
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import os
from arc_detector import ArcDetector, Point

def create_noisy_circle(center_x, center_y, radius, num_points=50, noise_amplitude=2.0):
    """Create a circle with zigzag noise"""
    points = []
    for i in range(num_points):
        angle = (i / num_points) * 2 * math.pi
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)

        # Add alternating noise (zigzag pattern)
        noise = noise_amplitude * (1 if i % 2 == 0 else -1)
        x += noise * math.cos(angle + math.pi/2)
        y += noise * math.sin(angle + math.pi/2)

        points.append((x, y))

    return points

def create_zigzag_line(start_x=0.0, end_x=240.0, num_points=80,
                       noise_amplitude=6.0, baseline_y=0.0, slope=0.0):
    """Create a straight line with pronounced alternating up/down noise"""
    xs = np.linspace(start_x, end_x, num_points)
    base_line = baseline_y + slope * (xs - start_x)

    # Alternating offsets exaggerate the zigzag so it is easy to see
    offsets = np.array([noise_amplitude if i % 2 == 0 else -noise_amplitude
                        for i in range(num_points)])
    ys = base_line + offsets

    noisy_points = list(zip(xs, ys))
    baseline_points = list(zip(xs, base_line))
    return noisy_points, baseline_points

def visualize_angle_changes(points, ax, is_first=False):
    """Visualize angle changes between consecutive segments"""
    points_obj = [Point(p[0], p[1]) for p in points]

    angle_changes = []
    positions = []

    for i in range(1, len(points_obj) - 1):
        v1 = Point(points_obj[i].x - points_obj[i-1].x, points_obj[i].y - points_obj[i-1].y)
        v2 = Point(points_obj[i+1].x - points_obj[i].x, points_obj[i+1].y - points_obj[i].y)

        len_v1 = math.sqrt(v1.x**2 + v1.y**2)
        len_v2 = math.sqrt(v2.x**2 + v2.y**2)

        if len_v1 < 1e-6 or len_v2 < 1e-6:
            continue

        v1_norm = Point(v1.x / len_v1, v1.y / len_v1)
        v2_norm = Point(v2.x / len_v2, v2.y / len_v2)

        # Cross product for direction
        cross = v1_norm.x * v2_norm.y - v1_norm.y * v2_norm.x

        # Dot product for angle
        dot = v1_norm.x * v2_norm.x + v1_norm.y * v2_norm.y
        dot = max(-1.0, min(1.0, dot))
        angle = math.degrees(math.acos(dot))

        # Store signed angle
        signed_angle = angle if cross >= 0 else -angle
        angle_changes.append(signed_angle)
        positions.append(i)

    # Plot angle changes
    colors = ['red' if a > 0 else 'blue' for a in angle_changes]
    ax.bar(positions, angle_changes, color=colors, alpha=0.6)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Point Index')
    ax.set_ylabel('Angle Change (degrees)')
    ax.set_title('Signed Angle Changes (Red=Left, Blue=Right)')
    ax.grid(True, alpha=0.3)

    # Calculate statistics
    sign_changes = sum(1 for i in range(len(angle_changes)-1)
                      if angle_changes[i] * angle_changes[i+1] < 0)
    alternation_ratio = sign_changes / (len(angle_changes) - 1) if len(angle_changes) > 1 else 0
    avg_abs_angle = sum(abs(a) for a in angle_changes) / len(angle_changes) if angle_changes else 0

    is_zigzag = alternation_ratio > 0.5 and avg_abs_angle > 0.5

    # Add statistics text with color-coded box
    stats_text = f"Sign changes: {sign_changes}/{len(angle_changes)-1}\n"
    stats_text += f"Alternation ratio: {alternation_ratio:.2f}\n"
    stats_text += f"Avg |angle|: {avg_abs_angle:.2f}°\n"
    stats_text += f"Zigzag detected: {is_zigzag}"

    # Color code the box for the first panel (original)
    if is_first:
        box_color = 'lightgreen' if is_zigzag else 'lightcoral'
        stats_text += f"\nStatus: {'✓ PASS' if is_zigzag else '✗ FAIL (expected zigzag)'}"
    else:
        box_color = 'wheat'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8),
            fontfamily='monospace', fontsize=9)

def visualize_radius_consistency(points, center, ax, title):
    """Visualize radius consistency"""
    points_obj = [Point(p[0], p[1]) for p in points]

    radii = [center.distance_to(p) for p in points_obj]
    avg_radius = sum(radii) / len(radii)

    ax.plot(range(len(radii)), radii, 'b-', linewidth=1, label='Radius to each point')
    ax.axhline(y=avg_radius, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_radius:.2f}')

    max_dev = max(abs(r - avg_radius) for r in radii)
    rel_dev = max_dev / avg_radius

    ax.fill_between(range(len(radii)),
                     avg_radius - max_dev,
                     avg_radius + max_dev,
                     alpha=0.2, color='red', label=f'Max deviation: ±{max_dev:.2f}')

    ax.set_xlabel('Point Index')
    ax.set_ylabel('Radius')
    ax.set_title(f'{title}\nRelative deviation: {rel_dev*100:.2f}%')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

def print_test_header():
    """Print standardized test header"""
    print("\n" + "="*80)
    print("SMOOTHING ALGORITHM TEST SUITE - Zigzag Detection & Multi-pass Smoothing")
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
        description='Visualize zigzag detection and smoothing algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python visualize_smoothing.py
  python visualize_smoothing.py --noise 3.0 --points 100
  python visualize_smoothing.py --smoothing-window 7 --no-smoothing
        '''
    )

    parser.add_argument('--center-x', type=float, default=100, help='Circle center X (default: 100)')
    parser.add_argument('--center-y', type=float, default=100, help='Circle center Y (default: 100)')
    parser.add_argument('--radius', type=float, default=50, help='Circle radius (default: 50)')
    parser.add_argument('--points', type=int, default=50, help='Number of points (default: 50)')
    parser.add_argument('--noise', type=float, default=2.5, help='Noise amplitude (default: 2.5)')
    parser.add_argument('--angle-tolerance', type=float, default=5.0, help='Angle tolerance in degrees (default: 5.0)')
    parser.add_argument('--radius-tolerance', type=float, default=0.02, help='Radius tolerance fraction (default: 0.02)')
    parser.add_argument('--min-arc-points', type=int, default=4, help='Minimum arc points (default: 4)')
    parser.add_argument('--smoothing-window', type=int, default=5, help='Smoothing window size (default: 5)')
    parser.add_argument('--no-smoothing', action='store_true', help='Disable smoothing')
    parser.add_argument('--dpi', type=int, default=150, help='Output DPI (default: 150)')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots (just save to files)')

    parser.add_argument('--merge_dist_threshold_multiplier', type=float, default=2.0, help='Multiplier for merge distance threshold')
    parser.add_argument('--merge_center_dist_threshold', type=float, default=0.1, help='Threshold for merging centers')
    parser.add_argument('--merge_radius_diff_threshold', type=float, default=0.1, help='Threshold for merging radii')
    parser.add_argument('--zigzag_len_epsilon', type=float, default=1e-6, help='Epsilon for zigzag length')
    parser.add_argument('--zigzag_alternation_ratio', type=float, default=0.5, help='Ratio for zigzag alternation')
    parser.add_argument('--zigzag_min_angle', type=float, default=2.0, help='Minimum angle for zigzag detection')
    parser.add_argument('--smoothing_lambda', type=float, default=0.4, help='Lambda for Taubin smoothing')
    parser.add_argument('--smoothing_mu', type=float, default=-0.42, help='Mu for Taubin smoothing')
    parser.add_argument('--smoothing_passes', type=int, default=6, help='Number of passes for Taubin smoothing')
    parser.add_argument('--curvature_cross_threshold', type=float, default=0.05, help='Threshold for curvature cross product')
    parser.add_argument('--min_radius', type=float, default=5.0, help='Minimum radius for arc detection')
    parser.add_argument('--full_circle_dist_threshold_multiplier', type=float, default=1.2, help='Multiplier for full circle distance threshold')
    parser.add_argument('--full_circle_angle_span', type=float, default=358.0, help='Minimum angle span for full circle')
    parser.add_argument('--least_squares_epsilon', type=float, default=1e-10, help='Epsilon for least squares fitting')
    parser.add_argument('--line-length', type=float, default=240.0, help='Length of the straight line example (default: 240)')
    parser.add_argument('--line-points', type=int, default=80, help='Number of points for straight line example (default: 80)')
    parser.add_argument('--line-noise', type=float, default=6.0, help='Noise amplitude for straight line zigzag (default: 6.0)')
    parser.add_argument('--line-slope', type=float, default=0.0, help='Optional slope for the straight line example (default: 0)')
    parser.add_argument('--line-zoom-window', type=int, default=14, help='Number of points to show in zoomed inset for straight line (default: 14)')

    args = parser.parse_args()

    total_passes = max(1, args.smoothing_passes)

    os.makedirs('output', exist_ok=True)

    # Print test header
    print_test_header()
    print(f"Algorithm: 3-pass Gaussian moving average smoothing")
    print(f"Test Case: Noisy circle with zigzag pattern")
    print(f"Parameters:")
    print(f"  - Circle: center=({args.center_x}, {args.center_y}), radius={args.radius}")
    print(f"  - Points: {args.points}")
    print(f"  - Noise amplitude: {args.noise}")
    print(f"  - Smoothing window: {args.smoothing_window}")
    print(f"  - Smoothing passes: {total_passes}")
    print(f"  - Smoothing enabled: {not args.no_smoothing}")

    # Create noisy circle
    center_x, center_y = args.center_x, args.center_y
    radius = args.radius
    noisy_points = create_noisy_circle(center_x, center_y, radius,
                                       num_points=args.points,
                                       noise_amplitude=args.noise)

    # Create detector
    detector = ArcDetector(
        angle_tolerance=args.angle_tolerance,
        radius_tolerance=args.radius_tolerance,
        min_arc_points=args.min_arc_points,
        enable_smoothing=not args.no_smoothing,
        smoothing_window=args.smoothing_window,
        merge_dist_threshold_multiplier=args.merge_dist_threshold_multiplier,
        merge_center_dist_threshold=args.merge_center_dist_threshold,
        merge_radius_diff_threshold=args.merge_radius_diff_threshold,
        zigzag_len_epsilon=args.zigzag_len_epsilon,
        zigzag_alternation_ratio=args.zigzag_alternation_ratio,
        zigzag_min_angle=args.zigzag_min_angle,
        smoothing_lambda=args.smoothing_lambda,
        smoothing_mu=args.smoothing_mu,
        smoothing_passes=total_passes,
        curvature_cross_threshold=args.curvature_cross_threshold,
        min_radius=args.min_radius,
        full_circle_dist_threshold_multiplier=args.full_circle_dist_threshold_multiplier,
        full_circle_angle_span=args.full_circle_angle_span,
        least_squares_epsilon=args.least_squares_epsilon
    )

    # Convert to Point objects
    points_obj = [Point(p[0], p[1]) for p in noisy_points]

    # Define EXPECTED results based on test parameters
    # With noise amplitude > 0, we EXPECT zigzag pattern to be detected
    # With smoothing enabled, we EXPECT radius consistency to improve
    expected = {
        'zigzag_detected': args.noise > 0,  # Expect zigzag if there's noise
        'improved_consistency': not args.no_smoothing,  # Expect improvement if smoothing enabled
        'description': 'Noisy circle should have zigzag pattern detected'
    }

    # Get ACTUAL results
    # Detect zigzag
    actual_zigzag = detector._detect_zigzag_pattern(points_obj)

    # Helper to get smoothed versions at different fractions of the total passes
    def smooth_polyline_with_passes(points_seq, pass_count):
        smoother = ArcDetector(
            enable_smoothing=True,
            smoothing_window=args.smoothing_window,
            smoothing_lambda=args.smoothing_lambda,
            smoothing_mu=args.smoothing_mu,
            smoothing_passes=pass_count
        )
        return smoother._smooth_polyline(points_seq)

    snapshot_specs = [
        (0.50, max(1, math.ceil(total_passes * 0.50))),
        (0.75, max(1, math.ceil(total_passes * 0.75))),
        (1.00, total_passes),
    ]

    smoothed_snapshots = []
    for pct, pass_count in snapshot_specs:
        pass_count = min(pass_count, total_passes)
        smoothed_points = smooth_polyline_with_passes(points_obj, pass_count)
        smoothed_snapshots.append({
            'pct': pct,
            'passes': pass_count,
            'points': smoothed_points
        })

    # Create a pronounced zigzag straight line to make the smoothing effect obvious
    zigzag_line, baseline_line = create_zigzag_line(
        start_x=0.0,
        end_x=args.line_length,
        num_points=args.line_points,
        noise_amplitude=args.line_noise,
        baseline_y=0.0,
        slope=args.line_slope
    )
    zigzag_line_obj = [Point(p[0], p[1]) for p in zigzag_line]
    line_smoothed_snapshots = []
    for pct, pass_count in snapshot_specs:
        pass_count = min(pass_count, total_passes)
        smoothed_line = smooth_polyline_with_passes(zigzag_line_obj, pass_count)
        line_smoothed_snapshots.append({
            'pct': pct,
            'passes': pass_count,
            'points': smoothed_line
        })

    # Calculate centers for radius analysis
    center = Point(center_x, center_y)

    # Calculate radius deviation improvement (ACTUAL)
    radii_original = [center.distance_to(p) for p in points_obj]
    avg_r_original = sum(radii_original) / len(radii_original)
    max_dev_original = max(abs(r - avg_r_original) for r in radii_original)
    rel_dev_original = max_dev_original / avg_r_original

    final_smoothed = smoothed_snapshots[-1]['points']

    radii_final = [center.distance_to(p) for p in final_smoothed]
    avg_r_final = sum(radii_final) / len(radii_final)
    max_dev_final = max(abs(r - avg_r_final) for r in radii_final)
    rel_dev_final = max_dev_final / avg_r_final

    # Store actual results
    actual_improved_consistency = rel_dev_final < rel_dev_original
    expected['original_deviation'] = rel_dev_original
    expected['final_deviation'] = rel_dev_final

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))

    # Row 1: Polylines
    ax1 = plt.subplot(3, 4, 1)
    x_vals = [p[0] for p in noisy_points]
    y_vals = [p[1] for p in noisy_points]
    ax1.plot(x_vals, y_vals, 'ro-', markersize=4, linewidth=1, label='Noisy')
    ax1.plot(center_x, center_y, 'gx', markersize=10, markeredgewidth=2, label='True center')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Original Noisy Polyline')

    colors = ['bo-', 'go-', 'mo-']
    for idx, snapshot in enumerate(smoothed_snapshots):
        ax_snapshot = plt.subplot(3, 4, idx + 2)
        xs = [p.x for p in snapshot['points']]
        ys = [p.y for p in snapshot['points']]
        ax_snapshot.plot(x_vals, y_vals, 'r.-', alpha=0.25, markersize=3, linewidth=0.5, label='Original')
        ax_snapshot.plot(xs, ys, colors[idx], markersize=4, linewidth=1,
                         label=f'After {snapshot["passes"]} passes')
        ax_snapshot.plot(center_x, center_y, 'gx', markersize=10, markeredgewidth=2)
        ax_snapshot.set_aspect('equal')
        ax_snapshot.grid(True, alpha=0.3)
        ax_snapshot.legend()
        pct_label = int(snapshot["pct"] * 100)
        ax_snapshot.set_title(f'After {pct_label}% of passes ({snapshot["passes"]})')

    # Row 2: Angle changes
    ax5 = plt.subplot(3, 4, 5)
    visualize_angle_changes(noisy_points, ax5, is_first=True)

    for idx, snapshot in enumerate(smoothed_snapshots):
        ax_angle = plt.subplot(3, 4, idx + 6)
        visualize_angle_changes([(p.x, p.y) for p in snapshot['points']], ax_angle)
        pct_label = int(snapshot["pct"] * 100)
        ax_angle.set_title(f'Angles after {pct_label}% ({snapshot["passes"]} passes)')

    # Row 3: Radius consistency
    ax9 = plt.subplot(3, 4, 9)
    visualize_radius_consistency(noisy_points, center, ax9, 'Original')

    for idx, snapshot in enumerate(smoothed_snapshots):
        ax_radius = plt.subplot(3, 4, idx + 10)
        pct_label = int(snapshot["pct"] * 100)
        title = f'{pct_label}% ({snapshot["passes"]} passes)'
        visualize_radius_consistency([(p.x, p.y) for p in snapshot['points']], center, ax_radius, title)

    # Build title with pass/fail status
    title = 'Zigzag Smoothing Algorithm - Step by Step Visualization'

    # Add pass/fail summary based on ACTUAL results matching EXPECTED
    zigzag_matches = (actual_zigzag == expected['zigzag_detected'])
    improvement_matches = (actual_improved_consistency == expected['improved_consistency'])

    zigzag_status = "✓ PASS" if zigzag_matches else "✗ FAIL"
    improvement_status = "✓ PASS" if improvement_matches else "✗ FAIL"

    title += f'\nZigzag Detection: {zigzag_status} | Radius Consistency Improved: {improvement_status}'
    title += f'\nDeviation: {expected["original_deviation"]*100:.2f}% → {expected["final_deviation"]*100:.2f}%'

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    plt.savefig('output/smoothing_visualization.png', dpi=args.dpi, bbox_inches='tight')
    print(f"✓ Saved: output/smoothing_visualization.png")

    # Straight-line focused visualization with exaggerated zigzag noise
    fig_line = plt.figure(figsize=(18, 10))

    line_x = [p[0] for p in zigzag_line]
    line_y = [p[1] for p in zigzag_line]
    base_line_y = [p[1] for p in baseline_line]
    line_snap_coords = []
    for snap in line_smoothed_snapshots:
        xs = [p.x for p in snap['points']]
        ys = [p.y for p in snap['points']]
        line_snap_coords.append({'xs': xs, 'ys': ys, 'snap': snap})
    final_line_points = line_smoothed_snapshots[-1]['points']

    # Panel 1: Highlight the zigzag pattern itself
    ax_line1 = plt.subplot(2, 2, 1)
    band_top = [b + args.line_noise for b in base_line_y]
    band_bottom = [b - args.line_noise for b in base_line_y]
    alternating_colors = ['#c44e52' if i % 2 == 0 else '#4c72b0' for i in range(len(line_x))]
    ax_line1.fill_between(line_x, band_bottom, band_top, color='#f2f2f2', alpha=0.9,
                          label=f'Zigzag band ±{args.line_noise}')
    ax_line1.plot(line_x, base_line_y, linestyle='--', color='#6c6c6c', linewidth=1.4, label='Ideal straight line')
    ax_line1.plot(line_x, line_y, color='#c44e52', linewidth=1.0, alpha=0.5, label='Noisy zigzag polyline')
    ax_line1.scatter(line_x, line_y, c=alternating_colors, s=26, edgecolors='black', linewidth=0.3,
                     label='Alternating vertices')
    ax_line1.set_title('Straight Line with Exaggerated Zigzag', fontsize=12)
    ax_line1.set_xlabel('X position')
    ax_line1.set_ylabel('Y position')
    ax_line1.grid(True, alpha=0.3)
    ax_line1.legend(loc='upper right', fontsize=8)

    # Panel 2: Full-line smoothing progression
    ax_line2 = plt.subplot(2, 2, 2)
    ax_line2.plot(line_x, base_line_y, linestyle='--', color='#6c6c6c', linewidth=1.2, label='Ideal straight line')
    ax_line2.plot(line_x, line_y, 'o-', color='#c44e52', markersize=3, linewidth=1, alpha=0.4, label='Original zigzag')
    colors_line = ['#4c72b0', '#55a868', '#8172b2']
    for idx, coord in enumerate(line_snap_coords):
        snap = coord['snap']
        pct_label = int(snap["pct"] * 100)
        label = f'After {pct_label}% ({snap["passes"]} passes)'
        color = colors_line[idx % len(colors_line)]
        ax_line2.plot(coord['xs'], coord['ys'], 'o-', color=color, markersize=3, linewidth=1 + 0.1 * idx, label=label)
    ax_line2.set_title('Smoothing Passes Overlaid (Straight Line)', fontsize=12)
    ax_line2.set_xlabel('X position')
    ax_line2.set_ylabel('Y position')
    ax_line2.grid(True, alpha=0.3)
    ax_line2.legend(loc='upper right', fontsize=8)

    # Panel 3: Zoomed-in view with movement arrows
    ax_line3 = plt.subplot(2, 2, 3)
    window_half = max(3, args.line_zoom_window // 2)
    zoom_center = len(line_x) // 2
    zoom_start = max(0, zoom_center - window_half)
    zoom_end = min(len(line_x), zoom_center + window_half)

    ax_line3.plot(line_x[zoom_start:zoom_end], base_line_y[zoom_start:zoom_end],
                  linestyle='--', color='#6c6c6c', linewidth=1.0, label='Ideal straight line')
    final_line_x = [p.x for p in final_line_points]
    final_line_y = [p.y for p in final_line_points]

    ax_line3.plot(line_x[zoom_start:zoom_end], line_y[zoom_start:zoom_end],
                  'o-', color='#c44e52', markersize=4, linewidth=1, alpha=0.5, label='Original zigzag')
    ax_line3.plot(final_line_x[zoom_start:zoom_end], final_line_y[zoom_start:zoom_end],
                  'o-', color='#55a868', markersize=4, linewidth=1.2, label='Final smoothed')

    # Show per-point motion from original to final in the zoom window
    zoom_dx = [final_line_points[i].x - zigzag_line_obj[i].x for i in range(zoom_start, zoom_end)]
    zoom_dy = [final_line_points[i].y - zigzag_line_obj[i].y for i in range(zoom_start, zoom_end)]
    ax_line3.quiver(line_x[zoom_start:zoom_end], line_y[zoom_start:zoom_end],
                    zoom_dx, zoom_dy, angles='xy', scale_units='xy', scale=1,
                    width=0.004, color='#ff7f0e', alpha=0.9, label='Movement to final')

    ax_line3.set_title(f'Zoomed Segment (points {zoom_start}-{zoom_end})', fontsize=12)
    ax_line3.set_xlabel('X position')
    ax_line3.set_ylabel('Y position')
    ax_line3.grid(True, alpha=0.3)
    ax_line3.legend(loc='upper right', fontsize=8)

    # Panel 4: Zigzag pattern as a 1D signal along the line
    ax_line4 = plt.subplot(2, 2, 4)
    indices = list(range(len(line_x)))
    baseline_minus = [b - args.line_noise for b in base_line_y]
    baseline_plus = [b + args.line_noise for b in base_line_y]
    ax_line4.fill_between(indices, baseline_minus, baseline_plus, color='#f2f2f2', alpha=0.9,
                          label='Zigzag band')
    ax_line4.plot(indices, line_y, 'o-', color='#c44e52', markersize=3, linewidth=1, alpha=0.5, label='Original zigzag')
    for idx, coord in enumerate(line_snap_coords):
        snap = coord['snap']
        pct_label = int(snap["pct"] * 100)
        color = colors_line[idx % len(colors_line)]
        label = f'{pct_label}% ({snap["passes"]} passes)'
        ax_line4.plot(indices, coord['ys'], 'o-', color=color, markersize=3, linewidth=1 + 0.1 * idx, label=label)
    ax_line4.set_title('Point-by-Point View (Y vs point index)', fontsize=12)
    ax_line4.set_xlabel('Point index along line')
    ax_line4.set_ylabel('Y offset')
    ax_line4.grid(True, alpha=0.3)
    ax_line4.legend(loc='upper right', fontsize=8)

    plt.suptitle('Straight Line Zigzag Smoothing — How Each Pass Reduces the Noise', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('output/smoothing_line_visualization.png', dpi=args.dpi, bbox_inches='tight')
    print(f"✓ Saved: output/smoothing_line_visualization.png")

    # Print test results
    test_results = []

    # Test 1: Zigzag detection
    details = {
        'description': expected['description'],
        'additional_info': f"Zigzag pattern {'detected' if actual_zigzag else 'not detected'} in noisy circle"
    }
    passed = print_test_result(
        "Zigzag Detection",
        "Zigzag detected" if expected['zigzag_detected'] else "No zigzag detected",
        "Zigzag detected" if actual_zigzag else "No zigzag detected",
        details
    )
    test_results.append(passed)

    # Test 2: Radius consistency improvement
    details = {
        'description': 'Smoothing should improve radius consistency',
        'additional_info': f"Deviation {'improved' if actual_improved_consistency else 'did not improve'} from {expected['original_deviation']*100:.2f}% to {expected['final_deviation']*100:.2f}% "
                          f"(change: {(expected['original_deviation'] - expected['final_deviation'])*100:.2f}%)"
    }
    passed = print_test_result(
        "Radius Consistency Improvement",
        "Improved" if expected['improved_consistency'] else "Not improved",
        "Improved" if actual_improved_consistency else "Not improved",
        details
    )
    test_results.append(passed)

    # Print summary
    total = len(test_results)
    passed_count = sum(test_results)
    failed = total - passed_count
    print_summary(total, passed_count, failed)

    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()
