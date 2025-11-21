#!/usr/bin/env python3
"""
Diagnostic script to analyze zigzag patterns in PDF polylines.
"""

import fitz
import argparse
import math
import sys
import numpy as np
from typing import List, Tuple, Dict
from arc_detector import ArcDetector, Point

def parse_point_string(s: str) -> Tuple[float, float]:
    """Parse "x,y" string to tuple"""
    try:
        x, y = map(float, s.split(','))
        return (x, y)
    except ValueError:
        return (0.0, 0.0)

class ZigzagDiagnoser:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.detector = ArcDetector() # Use defaults initially

    def extract_polylines(self, page_num: int = 0) -> List[List[Point]]:
        """Extract polylines from PDF page"""
        doc = fitz.open(self.pdf_path)
        if page_num >= len(doc):
            print(f"Error: Page {page_num} not found")
            return []
        
        page = doc[page_num]
        paths = page.get_drawings()
        
        all_polylines = []
        
        for path in paths:
            items = path.get('items', [])
            if not items:
                continue
                
            # Simple polyline extraction (similar to pdf_to_svg.py)
            current_polyline = []
            
            for item in items:
                if item[0] == 'l':
                    p1 = Point(item[1].x, item[1].y)
                    p2 = Point(item[2].x, item[2].y)
                    
                    if not current_polyline:
                        current_polyline = [p1, p2]
                    else:
                        # Check connectivity
                        if p1.distance_to(current_polyline[-1]) < 0.1:
                            current_polyline.append(p2)
                        else:
                            if len(current_polyline) >= 3:
                                all_polylines.append(current_polyline)
                            current_polyline = [p1, p2]
            
            if len(current_polyline) >= 3:
                all_polylines.append(current_polyline)
                
        return all_polylines

    def analyze_polyline(self, points: List[Point]) -> Dict:
        """Analyze a single polyline for zigzag characteristics"""
        if len(points) < 4:
            return None
            
        angle_changes = []
        
        for i in range(1, len(points) - 1):
            v1 = Point(points[i].x - points[i-1].x, points[i].y - points[i-1].y)
            v2 = Point(points[i+1].x - points[i].x, points[i+1].y - points[i].y)

            len_v1 = math.sqrt(v1.x**2 + v1.y**2)
            len_v2 = math.sqrt(v2.x**2 + v2.y**2)

            if len_v1 < 1e-6 or len_v2 < 1e-6:
                continue

            v1_norm = Point(v1.x / len_v1, v1.y / len_v1)
            v2_norm = Point(v2.x / len_v2, v2.y / len_v2)

            cross = v1_norm.x * v2_norm.y - v1_norm.y * v2_norm.x
            dot = v1_norm.x * v2_norm.x + v1_norm.y * v2_norm.y
            dot = max(-1.0, min(1.0, dot))
            angle = math.degrees(math.acos(dot))

            signed_angle = angle if cross >= 0 else -angle
            angle_changes.append(signed_angle)

        if len(angle_changes) < 3:
            return None

        # Calculate stats
        sign_changes = 0
        for i in range(len(angle_changes) - 1):
            if (angle_changes[i] * angle_changes[i+1]) < 0:
                sign_changes += 1
                
        alternation_ratio = sign_changes / (len(angle_changes) - 1)
        avg_abs_angle = sum(abs(a) for a in angle_changes) / len(angle_changes)
        max_abs_angle = max(abs(a) for a in angle_changes)
        
        # Check detection with actual detector
        is_zigzag_default = self.detector._detect_zigzag_pattern(points)
        
        return {
            'length': len(points),
            'avg_angle': avg_abs_angle,
            'max_angle': max_abs_angle,
            'alternation': alternation_ratio,
            'detected_default': is_zigzag_default
        }

    def run(self):
        print(f"Analyzing PDF: {self.pdf_path}")
        polylines = self.extract_polylines()
        print(f"Found {len(polylines)} polylines with >= 3 segments")
        
        zigzag_candidates = []
        detected_count = 0
        
        for i, poly in enumerate(polylines):
            stats = self.analyze_polyline(poly)
            if stats:
                # Filter for likely candidates (high alternation)
                if stats['alternation'] > 0.4:
                    zigzag_candidates.append((i, stats))
                if stats['detected_default']:
                    detected_count += 1
                    
        print(f"\nZigzag Detection Status:")
        print(f"  Total polylines analyzed: {len(polylines)}")
        print(f"  Detected with default settings (angle > 2.0°): {detected_count}")
        print(f"  Candidates with alternation > 0.4: {len(zigzag_candidates)}")
        
        if not zigzag_candidates:
            print("No zigzag patterns found.")
            return

        print("\nTop 10 Candidates (sorted by alternation ratio):")
        print(f"{'ID':<5} {'Len':<5} {'Alt Ratio':<10} {'Avg Angle':<10} {'Max Angle':<10} {'Detected?'}")
        print("-" * 60)
        
        # Sort by alternation ratio desc
        zigzag_candidates.sort(key=lambda x: x[1]['alternation'], reverse=True)
        
        angles = []
        
        print(f"\n--- High Alternation (>0.5) & Small Angle (<2.0°) Candidates ---")
        small_angle_count = 0
        for i, stats in zigzag_candidates:
            if stats['alternation'] > 0.5 and stats['avg_angle'] < 2.0:
                print(f"{i:<5} {stats['length']:<5} {stats['alternation']:.2f}       {stats['avg_angle']:.4f}°    {stats['max_angle']:.4f}°    {'YES' if stats['detected_default'] else 'NO'}")
                small_angle_count += 1
                angles.append(stats['avg_angle'])
        
        if small_angle_count == 0:
            print("No candidates found with high alternation and small angle (< 2.0°).")
        else:
            print(f"\nFound {small_angle_count} candidates with small angles that are currently MISSED.")
            
        print(f"\n--- Top 20 High Alternation Candidates (Any Angle) ---")
        for i, stats in zigzag_candidates[:20]:
            print(f"{i:<5} {stats['length']:<5} {stats['alternation']:.2f}       {stats['avg_angle']:.4f}°    {stats['max_angle']:.4f}°    {'YES' if stats['detected_default'] else 'NO'}")

        if angles:
            suggested_threshold = min(angles) if angles else 2.0
            print(f"\nAnalysis:")
            print(f"  Found {len(angles)} segments with high alternation but angle < 2.0°.")
            print(f"  Min angle in this group: {min(angles):.4f}°")
            print(f"  Avg angle in this group: {sum(angles)/len(angles):.4f}°")
            print(f"  Suggested zigzag_min_angle threshold: {suggested_threshold:.4f}°")
            
            if suggested_threshold < 2.0:
                print(f"  ACTION REQUIRED: Lower threshold from 2.0° to ~{suggested_threshold:.2f}° or lower.")
            else:
                print(f"  Current threshold (2.0°) seems appropriate.")

def main():
    parser = argparse.ArgumentParser(description='Diagnose zigzag detection in PDF')
    parser.add_argument('pdf_file', help='Input PDF file')
    args = parser.parse_args()
    
    diagnoser = ZigzagDiagnoser(args.pdf_file)
    diagnoser.run()

if __name__ == "__main__":
    main()
