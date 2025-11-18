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

def visualize_angle_changes(points, ax):
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

    # Add statistics text
    stats_text = f"Sign changes: {sign_changes}/{len(angle_changes)-1}\n"
    stats_text += f"Alternation ratio: {alternation_ratio:.2f}\n"
    stats_text += f"Avg |angle|: {avg_abs_angle:.2f}°\n"
    stats_text += f"Zigzag detected: {alternation_ratio > 0.5 and avg_abs_angle > 0.5}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
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

def main():
    # Create noisy circle
    center_x, center_y = 100, 100
    radius = 50
    noisy_points = create_noisy_circle(center_x, center_y, radius, num_points=50, noise_amplitude=2.5)

    # Create detector
    detector = ArcDetector(enable_smoothing=True, smoothing_window=5)

    # Convert to Point objects
    points_obj = [Point(p[0], p[1]) for p in noisy_points]

    # Detect zigzag
    is_zigzag = detector._detect_zigzag_pattern(points_obj)

    # Get smoothed versions for each pass
    smoothed_pass1 = detector._smooth_polyline(points_obj)

    # For multiple passes, we need to modify the detector temporarily
    detector_pass2 = ArcDetector(enable_smoothing=True, smoothing_window=5)
    smoothed_pass2 = detector_pass2._smooth_polyline(smoothed_pass1)

    detector_pass3 = ArcDetector(enable_smoothing=True, smoothing_window=5)
    smoothed_pass3 = detector_pass3._smooth_polyline(smoothed_pass2)

    # Calculate centers for radius analysis
    center = Point(center_x, center_y)

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

    ax2 = plt.subplot(3, 4, 2)
    x_smooth1 = [p.x for p in smoothed_pass1]
    y_smooth1 = [p.y for p in smoothed_pass1]
    ax2.plot(x_vals, y_vals, 'r.-', alpha=0.3, markersize=3, linewidth=0.5, label='Original')
    ax2.plot(x_smooth1, y_smooth1, 'bo-', markersize=4, linewidth=1, label='After pass 1')
    ax2.plot(center_x, center_y, 'gx', markersize=10, markeredgewidth=2)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('After Smoothing Pass 1')

    ax3 = plt.subplot(3, 4, 3)
    x_smooth2 = [p.x for p in smoothed_pass2]
    y_smooth2 = [p.y for p in smoothed_pass2]
    ax3.plot(x_vals, y_vals, 'r.-', alpha=0.2, markersize=3, linewidth=0.5, label='Original')
    ax3.plot(x_smooth2, y_smooth2, 'go-', markersize=4, linewidth=1, label='After pass 2')
    ax3.plot(center_x, center_y, 'gx', markersize=10, markeredgewidth=2)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_title('After Smoothing Pass 2')

    ax4 = plt.subplot(3, 4, 4)
    x_smooth3 = [p.x for p in smoothed_pass3]
    y_smooth3 = [p.y for p in smoothed_pass3]
    ax4.plot(x_vals, y_vals, 'r.-', alpha=0.2, markersize=3, linewidth=0.5, label='Original')
    ax4.plot(x_smooth3, y_smooth3, 'mo-', markersize=4, linewidth=1, label='After pass 3 (Final)')
    ax4.plot(center_x, center_y, 'gx', markersize=10, markeredgewidth=2)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_title('Final Smoothed Result')

    # Row 2: Angle changes
    ax5 = plt.subplot(3, 4, 5)
    visualize_angle_changes(noisy_points, ax5)

    ax6 = plt.subplot(3, 4, 6)
    visualize_angle_changes([(p.x, p.y) for p in smoothed_pass1], ax6)

    ax7 = plt.subplot(3, 4, 7)
    visualize_angle_changes([(p.x, p.y) for p in smoothed_pass2], ax7)

    ax8 = plt.subplot(3, 4, 8)
    visualize_angle_changes([(p.x, p.y) for p in smoothed_pass3], ax8)

    # Row 3: Radius consistency
    ax9 = plt.subplot(3, 4, 9)
    visualize_radius_consistency(noisy_points, center, ax9, 'Original')

    ax10 = plt.subplot(3, 4, 10)
    visualize_radius_consistency([(p.x, p.y) for p in smoothed_pass1], center, ax10, 'Pass 1')

    ax11 = plt.subplot(3, 4, 11)
    visualize_radius_consistency([(p.x, p.y) for p in smoothed_pass2], center, ax11, 'Pass 2')

    ax12 = plt.subplot(3, 4, 12)
    visualize_radius_consistency([(p.x, p.y) for p in smoothed_pass3], center, ax12, 'Pass 3 (Final)')

    plt.suptitle('Zigzag Smoothing Algorithm - Step by Step Visualization',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save
    plt.savefig('output/smoothing_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: output/smoothing_visualization.png")

    plt.show()

if __name__ == "__main__":
    main()
