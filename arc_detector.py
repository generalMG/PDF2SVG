#!/usr/bin/env python3
"""
Arc Detection from Polylines
Reconstructs circular arcs and circles from line segment approximations
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class Point:
    x: float
    y: float

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def to_tuple(self):
        return (self.x, self.y)


@dataclass
class Arc:
    center: Point
    radius: float
    start_angle: float  # degrees
    end_angle: float    # degrees
    points: List[Point]
    is_full_circle: bool = False

    def arc_length(self):
        """Calculate arc length"""
        angle_diff = abs(self.end_angle - self.start_angle)
        return (angle_diff / 360.0) * 2 * math.pi * self.radius


class ArcDetector:
    def __init__(self, angle_tolerance: float = 5.0, radius_tolerance: float = 0.02,
                 min_arc_points: int = 4, collinearity_threshold: float = 0.001,
                 enable_smoothing: bool = True, smoothing_window: int = 5,
                 min_arc_span: float = 15.0, min_segment_curvature: float = 10.0,
                 merge_dist_threshold_multiplier: float = 2.0,
                 merge_center_dist_threshold: float = 0.1,
                 merge_radius_diff_threshold: float = 0.1,
                 zigzag_len_epsilon: float = 1e-6,
                 zigzag_alternation_ratio: float = 0.5,
                 zigzag_min_angle: float = 2.0,
                 smoothing_lambda: float = 0.4,
                 smoothing_mu: float = -0.42,
                 smoothing_passes: int = 6,
                 curvature_cross_threshold: float = 0.05,
                 min_radius: float = 5.0,
                 full_circle_dist_threshold_multiplier: float = 1.2,
                 full_circle_angle_span: float = 358.0,
                 least_squares_epsilon: float = 1e-10):
        """
        Args:
            angle_tolerance: Max deviation in degrees from expected arc angles
            radius_tolerance: Max relative deviation in radius (0.02 = 2%)
            min_arc_points: Minimum points to consider as arc
            collinearity_threshold: Threshold for detecting straight lines
            enable_smoothing: Enable zigzag smoothing preprocessing
            smoothing_window: Window size for moving average smoothing (must be odd)
            min_arc_span: Minimum geometric arc span in degrees (default: 15.0)
            min_segment_curvature: Minimum cumulative curvature for segment validation in degrees (default: 10.0)
            merge_dist_threshold_multiplier: Multiplier for average segment length to determine adjacency (default: 2.0)
            merge_center_dist_threshold: Threshold for center distance relative to radius (default: 0.1)
            merge_radius_diff_threshold: Threshold for radius difference relative to radius (default: 0.1)
            zigzag_len_epsilon: Minimum vector length to consider for zigzag detection (default: 1e-6)
            zigzag_alternation_ratio: Ratio of sign changes to total angles for zigzag detection (default: 0.5)
            zigzag_min_angle: Minimum average absolute angle to consider for zigzag detection (default: 2.0)
            smoothing_lambda: Taubin smoothing lambda parameter (shrink) (default: 0.4)
            smoothing_mu: Taubin smoothing mu parameter (expand) (default: -0.42)
            smoothing_passes: Number of Taubin smoothing passes (default: 6)
            curvature_cross_threshold: Cross product threshold for curvature detection (default: 0.05)
            min_radius: Minimum radius to consider as a valid arc (default: 5.0)
            full_circle_dist_threshold_multiplier: Multiplier for average segment length to check loop closure (default: 1.2)
            full_circle_angle_span: Minimum angle span to consider as a full circle (default: 358.0)
            least_squares_epsilon: Epsilon for determinant check in least squares fit (default: 1e-10)
        """
        self.angle_tolerance = angle_tolerance
        self.radius_tolerance = radius_tolerance
        self.min_arc_points = min_arc_points
        self.collinearity_threshold = collinearity_threshold
        self.enable_smoothing = enable_smoothing
        self.smoothing_window = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1
        self.min_arc_span = min_arc_span
        self.min_segment_curvature = min_segment_curvature
        self.merge_dist_threshold_multiplier = merge_dist_threshold_multiplier
        self.merge_center_dist_threshold = merge_center_dist_threshold
        self.merge_radius_diff_threshold = merge_radius_diff_threshold
        self.zigzag_len_epsilon = zigzag_len_epsilon
        self.zigzag_alternation_ratio = zigzag_alternation_ratio
        self.zigzag_min_angle = zigzag_min_angle
        self.smoothing_lambda = smoothing_lambda
        self.smoothing_mu = smoothing_mu
        self.smoothing_passes = smoothing_passes
        self.curvature_cross_threshold = curvature_cross_threshold
        self.min_radius = min_radius
        self.full_circle_dist_threshold_multiplier = full_circle_dist_threshold_multiplier
        self.full_circle_angle_span = full_circle_angle_span
        self.least_squares_epsilon = least_squares_epsilon

    def detect_arcs(self, points: List[Tuple[float, float]]) -> List[Arc]:
        """
        Detect arcs from a sequence of connected points using
        Angle-Based Curvature Segmentation and Reconstruction (AASR)

        Returns list of Arc objects
        """
        if len(points) < self.min_arc_points:
            return []

        points = [Point(p[0], p[1]) for p in points]

        # Preprocessing: Smooth zigzag patterns if enabled
        if self.enable_smoothing and len(points) >= self.smoothing_window:
            if self._detect_zigzag_pattern(points):
                points = self._smooth_polyline(points)

        # Step 1: Segment polyline by curvature regions
        curved_segments = self._segment_by_curvature(points)

        # Step 2: Fit circles/arcs to curved segments
        arcs = []
        for segment in curved_segments:
            if len(segment) >= self.min_arc_points:
                arc = self._fit_arc_to_segment(segment)
                if arc:
                    arcs.append(arc)

        # Step 3: Merge adjacent arcs that are part of the same curve
        if len(arcs) > 1:
            arcs = self._merge_adjacent_arcs(arcs)

        return arcs

    def _merge_adjacent_arcs(self, arcs: List[Arc]) -> List[Arc]:
        """
        Merge adjacent arc segments that are part of the same curve

        This helps handle cases where a single curve gets split into multiple segments
        """
        if len(arcs) <= 1:
            return arcs

        merged = []
        current_arc = arcs[0]

        for next_arc in arcs[1:]:
            # Check if arcs are adjacent and have similar center/radius
            can_merge = False

            # Check if last point of current arc is close to first point of next arc
            if len(current_arc.points) > 0 and len(next_arc.points) > 0:
                dist = current_arc.points[-1].distance_to(next_arc.points[0])
                avg_segment_length = sum(
                    current_arc.points[i].distance_to(current_arc.points[i+1])
                    for i in range(len(current_arc.points)-1)
                ) / max(1, len(current_arc.points)-1)

                # Check if endpoints are adjacent (within threshold * average segment length)
                if dist < self.merge_dist_threshold_multiplier * avg_segment_length:
                    # Check if they have similar center and radius
                    center_dist = current_arc.center.distance_to(next_arc.center)
                    radius_diff = abs(current_arc.radius - next_arc.radius)
                    avg_radius = (current_arc.radius + next_arc.radius) / 2

                    if (center_dist < avg_radius * self.merge_center_dist_threshold and  # Centers within threshold of radius
                        radius_diff < avg_radius * self.merge_radius_diff_threshold):     # Radii within threshold
                        can_merge = True

            if can_merge:
                # Merge the arcs
                merged_points = current_arc.points + next_arc.points[1:]  # Skip duplicate point

                # Recalculate arc parameters
                avg_center = Point(
                    (current_arc.center.x + next_arc.center.x) / 2,
                    (current_arc.center.y + next_arc.center.y) / 2
                )
                avg_radius = (current_arc.radius + next_arc.radius) / 2

                # Use angles from first and last arc
                start_angle = current_arc.start_angle
                end_angle = next_arc.end_angle

                current_arc = Arc(
                    center=avg_center,
                    radius=avg_radius,
                    start_angle=start_angle,
                    end_angle=end_angle,
                    points=merged_points,
                    is_full_circle=False
                )
            else:
                # Can't merge, save current and move to next
                merged.append(current_arc)
                current_arc = next_arc

        # Don't forget the last arc
        merged.append(current_arc)

        return merged

    def _detect_zigzag_pattern(self, points: List[Point]) -> bool:
        """
        Detect if polyline exhibits zigzag pattern (alternating angle deviations)

        Returns True if zigzag pattern detected, False for smooth/straight lines
        """
        if len(points) < 4:
            return False

        # Calculate consecutive angle changes
        angle_changes = []
        for i in range(1, len(points) - 1):
            v1 = Point(points[i].x - points[i-1].x, points[i].y - points[i-1].y)
            v2 = Point(points[i+1].x - points[i].x, points[i+1].y - points[i].y)

            len_v1 = math.sqrt(v1.x**2 + v1.y**2)
            len_v2 = math.sqrt(v2.x**2 + v2.y**2)

            if len_v1 < self.zigzag_len_epsilon or len_v2 < self.zigzag_len_epsilon:
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

        if len(angle_changes) < 3:
            return False

        # Check for alternating pattern (zigzag signature)
        sign_changes = 0
        for i in range(len(angle_changes) - 1):
            # Count sign changes in consecutive angles
            if (angle_changes[i] * angle_changes[i+1]) < 0:
                sign_changes += 1

        # If more than 50% of angles alternate sign, it's a zigzag
        alternation_ratio = sign_changes / (len(angle_changes) - 1)

        # Also check if angles have significant magnitude (not just noise)
        avg_abs_angle = sum(abs(a) for a in angle_changes) / len(angle_changes)

        # Zigzag detected if: high alternation + meaningful angles
        # Increased threshold to 2.0° to only catch actual zigzag noise
        # High-resolution circles have ~1.5° changes, shouldn't trigger smoothing
        is_zigzag = alternation_ratio > self.zigzag_alternation_ratio and avg_abs_angle > self.zigzag_min_angle

        return is_zigzag

    def _calculate_overall_direction(self, points: List[Point]) -> Tuple[float, float]:
        """
        Calculate the overall direction vector of a polyline using linear regression

        Returns (dx, dy) normalized direction vector
        """
        if len(points) < 2:
            return (0.0, 0.0)

        # Simple approach: direction from first to last point
        # This gives the general trend
        dx = points[-1].x - points[0].x
        dy = points[-1].y - points[0].y

        length = math.sqrt(dx**2 + dy**2)
        if length < 1e-6:
            return (0.0, 0.0)

        return (dx / length, dy / length)

    def _smooth_polyline(self, points: List[Point]) -> List[Point]:
        """
        Smooth polyline using Taubin smoothing (volume-preserving)

        Alternates shrink and expand steps to remove noise without changing shape.
        This prevents the radius shrinkage problem of traditional moving averages.

        Args:
            points: List of Point objects

        Returns:
            Smoothed list of Point objects
        """
        if len(points) < 3:
            return points

        # Use adaptive window size based on number of points
        effective_window = min(self.smoothing_window, len(points) // 2)
        if effective_window % 2 == 0:
            effective_window += 1

        # Taubin smoothing parameters (fine-tuned for minimal distortion)
        lambda_smooth = self.smoothing_lambda    # Shrink coefficient (positive) - reduced for gentler smoothing
        mu_smooth = self.smoothing_mu      # Expand coefficient (negative, slightly larger magnitude)
        num_passes = self.smoothing_passes         # Even number ensures last pass is expand (counters shrinkage)

        smoothed = points
        half_window = effective_window // 2

        def mirror_index(idx: int, n: int) -> int:
            """
            Reflect index at boundaries to keep window size consistent and avoid edge drift.

            Example for n=5: indices ...-2,-1,0,1,2,3,4,5,6 -> 2,1,0,1,2,3,4,3,2
            """
            if idx < 0:
                return -idx
            if idx >= n:
                return (2 * n - 2) - idx
            return idx

        for pass_num in range(num_passes):
            # Alternate between shrink (lambda) and expand (mu)
            # pass 0: shrink, pass 1: expand, pass 2: shrink, pass 3: expand, pass 4: shrink, pass 5: expand
            coefficient = lambda_smooth if pass_num % 2 == 0 else mu_smooth
            new_smoothed = []

            for i in range(len(smoothed)):
                if i == 0 or i == len(smoothed) - 1:
                    # Keep endpoints unchanged
                    new_smoothed.append(smoothed[i])
                else:
                    # Use reflected indices to keep balanced window near edges
                    window_indices = [
                        mirror_index(j, len(smoothed))
                        for j in range(i - half_window, i + half_window + 1)
                    ]
                    window_points = [smoothed[j] for j in window_indices]

                    # Simple averaging (Laplacian operator)
                    avg_x = sum(p.x for p in window_points) / len(window_points)
                    avg_y = sum(p.y for p in window_points) / len(window_points)

                    # Taubin step: move toward average by coefficient
                    # Positive coefficient = shrink (move toward neighbors)
                    # Negative coefficient = expand (move away from neighbors)
                    new_x = smoothed[i].x + coefficient * (avg_x - smoothed[i].x)
                    new_y = smoothed[i].y + coefficient * (avg_y - smoothed[i].y)

                    new_smoothed.append(Point(new_x, new_y))

            smoothed = new_smoothed

        return smoothed

    def _segment_by_curvature(self, points: List[Point],
                              angle_tolerance_rad: float = None) -> List[List[Point]]:
        """
        Segment polyline into regions of consistent curvature using AASR algorithm

        Uses cumulative angle tracking to distinguish smooth curves from straight sections.
        Small instantaneous angles can indicate either straight lines OR smooth curves,
        so we track cumulative curvature to make the distinction.

        Returns list of curved segments (filters out straight sections)
        """
        if angle_tolerance_rad is None:
            # Convert angle tolerance from degrees to radians
            angle_tolerance_rad = math.radians(self.angle_tolerance)

        if len(points) < 3:
            return []

        segments = []
        current_segment = [points[0]]
        current_curvature_sign = None
        segment_cumulative_angle = 0.0  # Track total curvature in current segment

        for i in range(1, len(points) - 1):
            # Calculate vectors for consecutive segments
            v1 = Point(points[i].x - points[i-1].x, points[i].y - points[i-1].y)
            v2 = Point(points[i+1].x - points[i].x, points[i+1].y - points[i].y)

            # Normalize vectors
            len_v1 = math.sqrt(v1.x**2 + v1.y**2)
            len_v2 = math.sqrt(v2.x**2 + v2.y**2)

            if len_v1 < 1e-6 or len_v2 < 1e-6:
                # Skip degenerate segments
                continue

            v1_norm = Point(v1.x / len_v1, v1.y / len_v1)
            v2_norm = Point(v2.x / len_v2, v2.y / len_v2)

            # Calculate angle between vectors
            dot = v1_norm.x * v2_norm.x + v1_norm.y * v2_norm.y
            dot = max(-1.0, min(1.0, dot))  # Clamp to [-1, 1]
            theta = math.acos(dot)

            # Calculate curvature direction using cross product (2D)
            cross = v1_norm.x * v2_norm.y - v1_norm.y * v2_norm.x
            curvature_sign = 1 if cross > 0 else -1 if cross < 0 else 0

            # Determine if point is part of a curve based on:
            # 1. Cross product magnitude (curvature presence)
            # 2. Cumulative angle (total curvature, not instantaneous)
            # 3. Consistent curvature direction

            has_curvature = abs(cross) > self.curvature_cross_threshold  # Not straight (cross product threshold)

            if has_curvature:
                # Point has curvature - accumulate angle
                segment_cumulative_angle += theta

                if current_curvature_sign is None:
                    # Start new curved segment
                    current_curvature_sign = curvature_sign
                    current_segment.append(points[i])
                elif current_curvature_sign == curvature_sign:
                    # Continue current curved segment (same direction)
                    current_segment.append(points[i])
                else:
                    # Curvature direction changed - end current segment
                    # Include transition point in current segment before ending
                    current_segment.append(points[i])
                    # Use configurable threshold (default 10°, can be lowered to 5° for detecting smaller arcs)
                    if len(current_segment) >= self.min_arc_points and segment_cumulative_angle > math.radians(self.min_segment_curvature):
                        segments.append(current_segment)
                    # Start new segment with transition point
                    current_segment = [points[i]]
                    current_curvature_sign = curvature_sign
                    segment_cumulative_angle = theta  # Reset with current angle
            else:
                # Straight or nearly straight section
                # Use configurable threshold to determine if previous segment had enough curvature
                if segment_cumulative_angle > math.radians(self.min_segment_curvature):
                    # Previous segment had significant curvature, save it
                    if len(current_segment) >= self.min_arc_points:
                        segments.append(current_segment)
                # Reset for potential new segment
                current_segment = [points[i]]
                current_curvature_sign = None
                segment_cumulative_angle = 0.0

        # Add last point to current segment
        if len(points) > 0:
            current_segment.append(points[-1])

        # Add final segment if it's long enough and has enough curvature
        # Use configurable threshold
        if len(current_segment) >= self.min_arc_points and segment_cumulative_angle > math.radians(self.min_segment_curvature):
            segments.append(current_segment)

        return segments

    def _fit_circle_least_squares(self, points: List[Point]) -> Optional[Tuple[Point, float]]:
        """
        Fit circle to points using Kåsa algebraic least squares method

        More robust than 3-point geometric fit as it uses all points to
        find the optimal center and radius, averaging out noise.

        Args:
            points: List of Point objects

        Returns:
            Tuple of (center, radius) or None if fit fails
        """
        if len(points) < 3:
            return None

        # Extract coordinates
        n = len(points)
        x = [p.x for p in points]
        y = [p.y for p in points]

        # Calculate sums for Kåsa method
        sx = sum(x)
        sy = sum(y)
        sxx = sum(xi**2 for xi in x)
        syy = sum(yi**2 for yi in y)
        sxy = sum(xi*yi for xi, yi in zip(x, y))
        sxxx = sum(xi**3 for xi in x)
        syyy = sum(yi**3 for yi in y)
        sxyy = sum(xi*yi**2 for xi, yi in zip(x, y))
        sxxy = sum(xi**2*yi for xi, yi in zip(x, y))

        # Build linear system: A * [uc, vc] = b
        # where (uc, vc) are the circle center coordinates
        A = 2 * (sxx - sx*sx/n)
        B = 2 * (sxy - sx*sy/n)
        C = 2 * (sxy - sx*sy/n)
        D = 2 * (syy - sy*sy/n)

        E = sxxx + sxyy - (sx/n) * (sxx + syy)
        F = sxxy + syyy - (sy/n) * (sxx + syy)

        # Solve 2x2 system
        det = A * D - B * C
        if abs(det) < self.least_squares_epsilon:
            return None  # Singular system

        uc = (E * D - B * F) / det
        vc = (A * F - E * C) / det

        center = Point(uc, vc)

        # Calculate radius
        radii = [center.distance_to(p) for p in points]
        radius = sum(radii) / len(radii)

        return (center, radius)

    def _fit_arc_to_segment(self, segment: List[Point]) -> Optional[Arc]:
        """
        Fit a circle/arc to an entire curved segment using global least-squares

        Uses Kåsa least squares method for robust fitting, with fallback to
        3-point geometric fit and edge trimming if initial fit fails.

        This is the core AASR reconstruction step
        """
        if len(segment) < 3:
            return None

        # Check if segment is collinear (straight line, not a curve)
        if len(segment) >= 3 and self._are_collinear(segment[0], segment[1], segment[2]):
            return None

        # Try least squares fit first (uses all points, more robust)
        ls_result = self._fit_circle_least_squares(segment)

        if ls_result:
            center, avg_radius = ls_result
        else:
            # Fallback to 3-point geometric fit
            # Use interior points to avoid edge transition artifacts
            n = len(segment)
            if n >= 5:
                idx1 = max(0, n // 5)
                idx2 = n // 2
                idx3 = min(n - 1, (4 * n) // 5)
            else:
                idx1 = 0
                idx2 = n // 2
                idx3 = n - 1

            center = self._find_circle_center(segment[idx1], segment[idx2], segment[idx3])

            if center is None:
                return None

            radii = [center.distance_to(p) for p in segment]
            avg_radius = sum(radii) / len(radii)

        # Additional check: verify the segment has enough curvature
        # Calculate actual arc span instead of summing consecutive angle changes
        start_angle = self._calculate_angle(center, segment[0])
        end_angle = self._calculate_angle(center, segment[-1])
        arc_span = abs(self._calculate_angle_span(start_angle, end_angle))

        # Require minimum arc span (configurable, default 15°)
        # This correctly identifies smooth arcs that have small per-segment angles
        if arc_span < self.min_arc_span:
            return None

        # Filter out tiny arcs (increased threshold to reduce false positives)
        if avg_radius < self.min_radius:
            return None

        # Check radius consistency across all points (global fit quality)
        radii = [center.distance_to(p) for p in segment]
        max_deviation = max(abs(r - avg_radius) for r in radii)
        relative_deviation = max_deviation / avg_radius

        if relative_deviation > self.radius_tolerance:
            # Poor fit - try trimming edge points and re-fitting
            if len(segment) > self.min_arc_points + 2:
                # Try without first and last point
                trimmed = segment[1:-1]
                trimmed_result = self._fit_circle_least_squares(trimmed)

                if trimmed_result:
                    center_trimmed, radius_trimmed = trimmed_result
                    radii_trimmed = [center_trimmed.distance_to(p) for p in trimmed]
                    max_dev_trimmed = max(abs(r - radius_trimmed) for r in radii_trimmed)
                    rel_dev_trimmed = max_dev_trimmed / radius_trimmed

                    if rel_dev_trimmed <= self.radius_tolerance:
                        # Trimmed fit succeeded
                        start_angle = self._calculate_angle(center_trimmed, trimmed[0])
                        end_angle = self._calculate_angle(center_trimmed, trimmed[-1])

                        # Check arc span for trimmed segment
                        arc_span_trimmed = abs(self._calculate_angle_span(start_angle, end_angle))
                        if arc_span_trimmed < self.min_arc_span:
                            return None

                        return Arc(
                            center=center_trimmed,
                            radius=radius_trimmed,
                            start_angle=start_angle,
                            end_angle=end_angle,
                            points=trimmed,
                            is_full_circle=False
                        )

            # Still failed - use legacy fallback
            return self._try_fit_arc(segment)

        # Good fit! Arc parameters already calculated above (start_angle, end_angle)

        # CRITICAL: Validate arc direction - ensure arc contains the actual points
        # For a 270° arc, we should detect the 270° segment with points, not the empty 90°
        arc_span = self._calculate_angle_span(start_angle, end_angle)

        # Check if points are actually on this arc or its complement
        # Calculate angles for all points and check distribution
        point_angles = [self._calculate_angle(center, p) for p in segment]

        # Determine if points follow the arc from start_angle to end_angle
        # or if they're on the complementary arc
        points_on_arc = self._validate_arc_direction(
            point_angles, start_angle, end_angle, arc_span
        )

        # If points are on the complementary arc, swap start/end
        if not points_on_arc:
            # Points are on the opposite side - use complementary arc
            start_angle, end_angle = end_angle, start_angle
            arc_span = 360 - arc_span

        # Check if it's a full circle
        is_full_circle = False
        if len(segment) >= 8:
            first_to_last_dist = segment[0].distance_to(segment[-1])
            avg_segment_length = sum(segment[i].distance_to(segment[i+1])
                                    for i in range(len(segment)-1)) / (len(segment)-1)

            if first_to_last_dist < self.full_circle_dist_threshold_multiplier * avg_segment_length:
                if arc_span >= self.full_circle_angle_span:
                    is_full_circle = True

        return Arc(
            center=center,
            radius=avg_radius,
            start_angle=start_angle,
            end_angle=end_angle,
            points=segment,
            is_full_circle=is_full_circle
        )

    def detect_arcs_legacy(self, points: List[Tuple[float, float]]) -> List[Arc]:
        """
        Legacy arc detection method (sliding window approach)
        Kept for compatibility/fallback
        """
        if len(points) < self.min_arc_points:
            return []

        points = [Point(p[0], p[1]) for p in points]
        arcs = []
        i = 0

        while i < len(points) - self.min_arc_points + 1:
            arc = self._try_fit_arc(points[i:])
            if arc and len(arc.points) >= self.min_arc_points:
                # Filter out tiny arcs (likely noise or very small features)
                # Minimum radius of 5.0 units to avoid detecting micro-circles
                if arc.radius >= self.min_radius:
                    arcs.append(arc)
                    # Advance by the full arc length to prevent overlapping detections
                    i += len(arc.points)
                else:
                    i += 1
            else:
                i += 1

        return arcs

    def _try_fit_arc(self, points: List[Point], max_points: int = None) -> Optional[Arc]:
        """
        Try to fit an arc starting from the beginning of points
        Returns Arc if successful, None otherwise
        """
        if max_points is None:
            max_points = len(points)

        # Need at least 3 points to define an arc
        if len(points) < 3:
            return None

        # Check if first 3 points are collinear (not an arc)
        if self._are_collinear(points[0], points[1], points[2]):
            return None

        # Find circle through first 3 points
        center = self._find_circle_center(points[0], points[1], points[2])
        if center is None:
            return None

        radius = center.distance_to(points[0])

        # Check for degenerate case (zero or very small radius)
        if radius < 1e-6:
            return None

        # Try to extend the arc as far as possible
        arc_points = [points[0], points[1], points[2]]

        for i in range(3, min(len(points), max_points)):
            # Check if this point lies on the circle
            dist = center.distance_to(points[i])
            relative_error = abs(dist - radius) / radius

            if relative_error > self.radius_tolerance:
                break

            # Check angular consistency (points should be evenly spaced)
            if not self._check_angular_consistency(center, arc_points[-2:] + [points[i]]):
                break

            arc_points.append(points[i])

        if len(arc_points) < self.min_arc_points:
            return None

        # Calculate arc parameters
        start_angle = self._calculate_angle(center, arc_points[0])
        end_angle = self._calculate_angle(center, arc_points[-1])

        # Check if it's a full circle (end near start)
        is_full_circle = False
        if len(arc_points) >= 8:  # At least 8 points for a circle
            first_to_last_dist = arc_points[0].distance_to(arc_points[-1])
            avg_segment_length = sum(arc_points[i].distance_to(arc_points[i+1])
                                    for i in range(len(arc_points)-1)) / (len(arc_points)-1)

            # If first and last points are very close (within 1.2 segment lengths)
            # Use stricter threshold to avoid misclassifying partial arcs
            if first_to_last_dist < self.full_circle_dist_threshold_multiplier * avg_segment_length:
                # Check if arc spans very close to 360 degrees
                # Must be at least 358 degrees to be considered a full circle
                angle_span = self._calculate_angle_span(start_angle, end_angle)
                if angle_span >= self.full_circle_angle_span:  # Very strict: must cover almost full circle
                    is_full_circle = True

        return Arc(
            center=center,
            radius=radius,
            start_angle=start_angle,
            end_angle=end_angle,
            points=arc_points,
            is_full_circle=is_full_circle
        )

    def _validate_arc_direction(self, point_angles: List[float],
                                 start_angle: float, end_angle: float,
                                 arc_span: float) -> bool:
        """
        Validate that points are actually on the arc from start_angle to end_angle,
        not on the complementary arc.

        This fixes the 270° arc bug where the algorithm detected the empty 90° segment
        instead of the 270° segment containing the points.

        Args:
            point_angles: Angles of all points relative to center
            start_angle: Start angle of proposed arc
            end_angle: End angle of proposed arc
            arc_span: Span of proposed arc (degrees)

        Returns:
            True if points are on the arc, False if on complementary arc
        """
        if len(point_angles) < 2:
            return True

        # Count how many points fall within the arc span
        points_in_arc = 0
        points_outside_arc = 0

        for angle in point_angles:
            # Normalize angle to be in range for comparison
            # Check if angle is between start and end (going counterclockwise)
            if self._angle_in_range(angle, start_angle, end_angle):
                points_in_arc += 1
            else:
                points_outside_arc += 1

        # Points should be on the arc, not outside it
        # If more points are outside, we detected the complementary arc
        return points_in_arc > points_outside_arc

    def _angle_in_range(self, angle: float, start: float, end: float) -> bool:
        """
        Check if angle is in the range from start to end (counterclockwise).

        Handles wraparound at 360°/0°.
        """
        # Normalize all angles to [0, 360)
        angle = angle % 360
        start = start % 360
        end = end % 360

        if start <= end:
            # Simple case: no wraparound
            return start <= angle <= end
        else:
            # Wraparound case: e.g., start=350°, end=10°
            return angle >= start or angle <= end

    def _are_collinear(self, p1: Point, p2: Point, p3: Point) -> bool:
        """Check if three points are collinear using cross product"""
        # Cross product of vectors (p2-p1) and (p3-p1)
        cross = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)

        # Calculate area of triangle formed by the points
        area = abs(cross) / 2.0

        # Calculate perimeter
        d12 = p1.distance_to(p2)
        d23 = p2.distance_to(p3)
        d31 = p3.distance_to(p1)
        perimeter = d12 + d23 + d31

        if perimeter == 0:
            return True

        # Normalized measure: ratio of area to perimeter squared
        normalized_area = area / (perimeter ** 2)

        return normalized_area < self.collinearity_threshold

    def _find_circle_center(self, p1: Point, p2: Point, p3: Point) -> Optional[Point]:
        """
        Find center of circle passing through three points
        Uses perpendicular bisector method
        """
        # Calculate midpoints
        mid12 = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
        mid23 = Point((p2.x + p3.x) / 2, (p2.y + p3.y) / 2)

        # Calculate slopes of chords
        dx12 = p2.x - p1.x
        dy12 = p2.y - p1.y
        dx23 = p3.x - p2.x
        dy23 = p3.y - p2.y

        # Check for degenerate cases (duplicate points)
        if abs(dx12) < 1e-10 and abs(dy12) < 1e-10:
            return None  # p1 and p2 are the same
        if abs(dx23) < 1e-10 and abs(dy23) < 1e-10:
            return None  # p2 and p3 are the same

        # Handle vertical chords
        if abs(dx12) < 1e-10:
            if abs(dx23) < 1e-10:
                return None  # Both vertical, collinear
            # First chord vertical, perpendicular is horizontal
            cx = mid12.x
            # Perpendicular slope of second chord
            if abs(dy23) < 1e-10:
                return None  # Second chord is horizontal, collinear with perpendicular
            slope23_perp = -dx23 / dy23
            cy = mid23.y + slope23_perp * (cx - mid23.x)
            return Point(cx, cy)

        if abs(dx23) < 1e-10:
            # Second chord vertical, perpendicular is horizontal
            cx = mid23.x
            # Perpendicular slope of first chord
            if abs(dy12) < 1e-10:
                return None  # First chord is horizontal, collinear with perpendicular
            slope12_perp = -dx12 / dy12
            cy = mid12.y + slope12_perp * (cx - mid12.x)
            return Point(cx, cy)

        # Check for horizontal chords
        if abs(dy12) < 1e-10:
            # First chord is horizontal, perpendicular is vertical
            if abs(dy23) < 1e-10:
                return None  # Both horizontal, collinear
            cy = mid12.y
            slope23_perp = -dx23 / dy23
            cx = mid23.x + (cy - mid23.y) / slope23_perp
            return Point(cx, cy)

        if abs(dy23) < 1e-10:
            # Second chord is horizontal, perpendicular is vertical
            cy = mid23.y
            slope12_perp = -dx12 / dy12
            cx = mid12.x + (cy - mid12.y) / slope12_perp
            return Point(cx, cy)

        # Calculate perpendicular slopes
        slope12_perp = -dx12 / dy12
        slope23_perp = -dx23 / dy23

        # Check if perpendiculars are parallel (points are collinear)
        if abs(slope12_perp - slope23_perp) < 1e-10:
            return None

        # Find intersection of perpendicular bisectors
        # Line 1: y - mid12.y = slope12_perp * (x - mid12.x)
        # Line 2: y - mid23.y = slope23_perp * (x - mid23.x)

        cx = ((mid23.y - mid12.y) + slope12_perp * mid12.x - slope23_perp * mid23.x) / \
             (slope12_perp - slope23_perp)
        cy = mid12.y + slope12_perp * (cx - mid12.x)

        return Point(cx, cy)

    def _calculate_angle(self, center: Point, point: Point) -> float:
        """Calculate angle from center to point in degrees [0, 360)"""
        angle = math.degrees(math.atan2(point.y - center.y, point.x - center.x))
        return angle % 360

    def _calculate_angle_span(self, start_angle: float, end_angle: float) -> float:
        """Calculate the angular span of an arc"""
        diff = (end_angle - start_angle) % 360
        return diff

    def _check_angular_consistency(self, center: Point, points: List[Point]) -> bool:
        """
        Check if points are angularly consistent (roughly evenly spaced on arc)
        """
        if len(points) < 2:
            return True

        angles = [self._calculate_angle(center, p) for p in points]

        # Calculate angular differences
        angle_diffs = []
        for i in range(len(angles) - 1):
            diff = (angles[i+1] - angles[i]) % 360
            # Handle wraparound (e.g., 359° to 1°)
            if diff > 180:
                diff = diff - 360
            angle_diffs.append(abs(diff))

        if not angle_diffs:
            return True

        # Check if angular differences are consistent
        avg_diff = sum(angle_diffs) / len(angle_diffs)

        if avg_diff < 0.1:  # Too small, might be noise
            return False

        for diff in angle_diffs:
            if abs(diff - avg_diff) > self.angle_tolerance:
                return False

        return True

    def is_closed_loop(self, points: List[Tuple[float, float]],
                       tolerance_factor: float = 1.5) -> bool:
        """
        Check if polyline forms a closed loop (first point ≈ last point)

        Args:
            points: List of (x, y) points
            tolerance_factor: Multiplier for average segment length to determine closure

        Returns:
            True if first and last points are close enough
        """
        if len(points) < self.min_arc_points:
            return False

        # Calculate average segment length
        total_length = 0
        for i in range(len(points) - 1):
            dx = points[i+1][0] - points[i][0]
            dy = points[i+1][1] - points[i][1]
            total_length += math.sqrt(dx**2 + dy**2)

        avg_segment_length = total_length / (len(points) - 1)

        # Check distance between first and last point
        dx = points[-1][0] - points[0][0]
        dy = points[-1][1] - points[0][1]
        closure_dist = math.sqrt(dx**2 + dy**2)

        return closure_dist < tolerance_factor * avg_segment_length

    def check_radius_consistency(self, points: List[Tuple[float, float]]) -> Optional[Tuple[Point, float, float]]:
        """
        Check if points lie on a circle with consistent radius

        Args:
            points: List of (x, y) points

        Returns:
            (center, radius, relative_deviation) if consistent, None otherwise
        """
        if len(points) < 3:
            return None

        points_obj = [Point(p[0], p[1]) for p in points]

        # Calculate approximate center (centroid)
        cx = sum(p.x for p in points_obj) / len(points_obj)
        cy = sum(p.y for p in points_obj) / len(points_obj)
        center = Point(cx, cy)

        # Calculate distances from center
        radii = [center.distance_to(p) for p in points_obj]
        avg_radius = sum(radii) / len(radii)

        # Check consistency
        max_deviation = max(abs(r - avg_radius) for r in radii)
        relative_deviation = max_deviation / avg_radius if avg_radius > 0 else float('inf')

        return center, avg_radius, relative_deviation

    def detect_circle_global(self, points: List[Tuple[float, float]]) -> Optional[Arc]:
        """
        Detect if polyline is a complete circle using global analysis

        This is a fast preprocessing step for obvious circles before AASR.

        Args:
            points: List of (x, y) points

        Returns:
            Arc object if circle detected, None otherwise
        """
        if len(points) < self.min_arc_points:
            return None

        # Convert to Point objects for preprocessing
        points_obj = [Point(p[0], p[1]) for p in points]

        # Apply smoothing if zigzag detected
        if self.enable_smoothing and len(points_obj) >= self.smoothing_window:
            if self._detect_zigzag_pattern(points_obj):
                points_obj = self._smooth_polyline(points_obj)
                # Convert back to tuples for closed loop check
                points = [(p.x, p.y) for p in points_obj]

        # Check if it's a closed loop
        if not self.is_closed_loop(points, tolerance_factor=self.full_circle_dist_threshold_multiplier):
            return None

        # Check radius consistency
        result = self.check_radius_consistency(points)
        if result is None:
            return None

        center, radius, rel_deviation = result

        # Must have consistent radius
        if rel_deviation > self.radius_tolerance:
            return None

        # Filter out tiny circles (likely noise)
        if radius < self.min_radius:
            return None

        # Create full circle arc
        return Arc(
            center=center,
            radius=radius,
            start_angle=0,
            end_angle=360,
            points=points_obj,
            is_full_circle=True
        )

    def classify_arc(self, arc: Arc) -> str:
        """Classify arc type: full_circle, major_arc, minor_arc, semicircle"""
        if arc.is_full_circle:
            return "full_circle"

        angle_span = self._calculate_angle_span(arc.start_angle, arc.end_angle)

        if abs(angle_span - 180) < 5:
            return "semicircle"
        elif angle_span > 180:
            return "major_arc"
        else:
            return "minor_arc"


def extract_polylines_from_paths(paths: List[Dict[str, Any]]) -> List[List[Tuple[float, float]]]:
    """
    Extract polylines (sequences of connected line segments) from PDF paths
    """
    polylines = []

    for path in paths:
        items = path.get('items', [])
        if not items:
            continue

        current_polyline = []

        for item in items:
            if item['type'] == 'line':
                from_point = item['from']
                to_point = item['to']

                # Parse point strings like "Point(x, y)"
                if isinstance(from_point, str):
                    from_point = parse_point_string(from_point)
                if isinstance(to_point, str):
                    to_point = parse_point_string(to_point)

                if not current_polyline:
                    current_polyline.append(from_point)
                current_polyline.append(to_point)

        if len(current_polyline) >= 3:  # At least 3 points for potential arc
            polylines.append(current_polyline)

    return polylines


def parse_point_string(point_str: str) -> Tuple[float, float]:
    """Parse 'Point(x, y)' string to (x, y) tuple"""
    if isinstance(point_str, (list, tuple)):
        return tuple(point_str[:2])

    # Remove 'Point(' prefix and ')' suffix
    point_str = point_str.replace('Point(', '').replace(')', '')
    parts = point_str.split(',')
    return (float(parts[0].strip()), float(parts[1].strip()))


if __name__ == "__main__":
    # Test with sample circle/arc data
    print("Testing Arc Detection Algorithm")
    print("=" * 80)

    # Test 1: Perfect circle (24 points)
    print("\nTest 1: Perfect Circle (24 points)")
    center_x, center_y = 100, 100
    radius = 50
    circle_points = []
    for i in range(24):
        angle = (i / 24) * 2 * math.pi
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        circle_points.append((x, y))

    detector = ArcDetector(angle_tolerance=10.0, radius_tolerance=0.05, min_arc_points=5)
    arcs = detector.detect_arcs(circle_points)

    print(f"Detected {len(arcs)} arc(s)")
    for i, arc in enumerate(arcs):
        print(f"  Arc {i+1}:")
        print(f"    Center: ({arc.center.x:.2f}, {arc.center.y:.2f})")
        print(f"    Radius: {arc.radius:.2f}")
        print(f"    Angles: {arc.start_angle:.1f}° to {arc.end_angle:.1f}°")
        print(f"    Points: {len(arc.points)}")
        print(f"    Type: {detector.classify_arc(arc)}")
        print(f"    Arc Length: {arc.arc_length():.2f}")

    # Test 2: 90-degree arc
    print("\nTest 2: 90-degree Arc")
    arc_points = []
    for i in range(7):
        angle = (i / 6) * (math.pi / 2)  # 0 to 90 degrees
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        arc_points.append((x, y))

    arcs = detector.detect_arcs(arc_points)
    print(f"Detected {len(arcs)} arc(s)")
    for i, arc in enumerate(arcs):
        print(f"  Arc {i+1}:")
        print(f"    Center: ({arc.center.x:.2f}, {arc.center.y:.2f})")
        print(f"    Radius: {arc.radius:.2f}")
        print(f"    Angles: {arc.start_angle:.1f}° to {arc.end_angle:.1f}°")
        print(f"    Type: {detector.classify_arc(arc)}")

    # Test 3: Straight line (should not detect arc)
    print("\nTest 3: Straight Line (should not detect)")
    line_points = [(0, 0), (10, 10), (20, 20), (30, 30)]
    arcs = detector.detect_arcs(line_points)
    print(f"Detected {len(arcs)} arc(s) - Expected: 0")
