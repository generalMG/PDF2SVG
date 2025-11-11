#!/usr/bin/env python3
"""
PDF Drawing Analyzer - Extract and analyze vector graphics from PDF files
Focuses on identifying curves, arcs, and their geometric parameters
"""

import fitz  # PyMuPDF
import json
from pathlib import Path
from typing import Dict, List, Any
import math


def analyze_pdf_page(page: fitz.Page) -> Dict[str, Any]:
    """Extract detailed vector graphics information from a PDF page"""

    # Get the raw drawing commands
    paths = page.get_drawings()

    analysis = {
        "page_number": page.number,
        "page_size": {"width": page.rect.width, "height": page.rect.height},
        "total_paths": len(paths),
        "path_details": [],
        "curve_types": {},
        "statistics": {
            "lines": 0,
            "curves": 0,
            "bezier_curves": 0,
            "rectangles": 0,
            "quads": 0
        }
    }

    for i, path in enumerate(paths):
        path_info = {
            "path_id": i,
            "items": [],
            "bbox": path.get("rect"),
            "fill_color": path.get("fill"),
            "stroke_color": path.get("color"),
            "width": path.get("width")
        }

        # Analyze each item in the path
        for item in path.get("items", []):
            item_type = item[0]  # 'l' for line, 'c' for curve, 're' for rectangle, 'qu' for quad

            if item_type == "l":  # Line
                analysis["statistics"]["lines"] += 1
                path_info["items"].append({
                    "type": "line",
                    "from": item[1],
                    "to": item[2]
                })

            elif item_type == "c":  # Cubic Bezier curve
                analysis["statistics"]["bezier_curves"] += 1
                analysis["statistics"]["curves"] += 1
                path_info["items"].append({
                    "type": "cubic_bezier",
                    "start": item[1],
                    "control1": item[2],
                    "control2": item[3],
                    "end": item[4]
                })

            elif item_type == "re":  # Rectangle
                analysis["statistics"]["rectangles"] += 1
                path_info["items"].append({
                    "type": "rectangle",
                    "rect": item[1]
                })

            elif item_type == "qu":  # Quadratic curve
                analysis["statistics"]["quads"] += 1
                analysis["statistics"]["curves"] += 1
                path_info["items"].append({
                    "type": "quadratic_bezier",
                    "start": item[1],
                    "control": item[2],
                    "end": item[3]
                })

        if path_info["items"]:
            analysis["path_details"].append(path_info)

    return analysis


def detect_arcs_from_bezier(bezier_data: Dict[str, Any], tolerance: float = 0.01) -> Dict[str, Any]:
    """
    Attempt to detect if a Bezier curve represents a circular arc
    Returns arc parameters if detected
    """
    if bezier_data["type"] == "cubic_bezier":
        p0 = bezier_data["start"]
        p1 = bezier_data["control1"]
        p2 = bezier_data["control2"]
        p3 = bezier_data["end"]

        # Check if this might be a circular arc by analyzing control points
        # This is a simplified heuristic - real arc detection is more complex

        # Calculate distances
        d01 = math.sqrt((p1.x - p0.x)**2 + (p1.y - p0.y)**2)
        d23 = math.sqrt((p3.x - p2.x)**2 + (p3.y - p2.y)**2)
        d03 = math.sqrt((p3.x - p0.x)**2 + (p3.y - p0.y)**2)

        # For circular arcs, control points follow specific patterns
        # This is a basic check
        if abs(d01 - d23) < tolerance * max(d01, d23):
            return {
                "is_arc": True,
                "chord_length": d03,
                "control_distance": d01
            }

    return {"is_arc": False}


def analyze_pdf_file(pdf_path: str, detailed: bool = False) -> Dict[str, Any]:
    """Analyze a complete PDF file"""

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return {"error": f"File not found: {pdf_path}"}

    doc = fitz.open(str(pdf_path))

    file_analysis = {
        "file_name": pdf_path.name,
        "file_path": str(pdf_path),
        "page_count": len(doc),
        "pages": []
    }

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_analysis = analyze_pdf_page(page)

        if not detailed:
            # Remove detailed path info for summary
            page_analysis.pop("path_details", None)

        file_analysis["pages"].append(page_analysis)

    doc.close()

    # Calculate overall statistics
    file_analysis["overall_statistics"] = {
        "total_lines": sum(p["statistics"]["lines"] for p in file_analysis["pages"]),
        "total_curves": sum(p["statistics"]["curves"] for p in file_analysis["pages"]),
        "total_bezier_curves": sum(p["statistics"]["bezier_curves"] for p in file_analysis["pages"]),
        "total_rectangles": sum(p["statistics"]["rectangles"] for p in file_analysis["pages"]),
        "total_quads": sum(p["statistics"]["quads"] for p in file_analysis["pages"])
    }

    return file_analysis


def extract_raw_commands(pdf_path: str, page_num: int = 0) -> str:
    """Extract raw PDF commands to see low-level drawing operations"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # Get the raw page content stream
    xref = page.get_contents()

    if isinstance(xref, list):
        # Multiple content streams
        commands = []
        for x in xref:
            commands.append(doc.xref_stream(x).decode('latin-1', errors='ignore'))
        return "\n".join(commands)
    else:
        return doc.xref_stream(xref).decode('latin-1', errors='ignore')


if __name__ == "__main__":
    import sys

    # Test with provided PDF files
    pdf_files = [
        "/mnt/d/mg_ai_research/workspace/cadAI/pdf2cad2/pdf_files/60355K178_Ball Bearing.pdf",
        "/mnt/d/mg_ai_research/workspace/cadAI/pdf2cad2/pdf_files/61355K31_Combination Clutch Brake.pdf"
    ]

    for pdf_file in pdf_files:
        print(f"\n{'='*80}")
        print(f"Analyzing: {Path(pdf_file).name}")
        print(f"{'='*80}")

        # Summary analysis
        analysis = analyze_pdf_file(pdf_file, detailed=False)

        if "error" in analysis:
            print(f"Error: {analysis['error']}")
            continue

        print(f"\nFile: {analysis['file_name']}")
        print(f"Pages: {analysis['page_count']}")
        print(f"\nOverall Statistics:")
        print(f"  Lines: {analysis['overall_statistics']['total_lines']}")
        print(f"  Curves (total): {analysis['overall_statistics']['total_curves']}")
        print(f"  Bezier Curves: {analysis['overall_statistics']['total_bezier_curves']}")
        print(f"  Quadratic Curves: {analysis['overall_statistics']['total_quads']}")
        print(f"  Rectangles: {analysis['overall_statistics']['total_rectangles']}")

        print(f"\nPer-Page Breakdown:")
        for page_info in analysis['pages']:
            print(f"  Page {page_info['page_number']}: "
                  f"{page_info['total_paths']} paths, "
                  f"{page_info['statistics']['lines']} lines, "
                  f"{page_info['statistics']['curves']} curves")

        # Save detailed analysis to JSON
        output_file = Path(pdf_file).stem + "_analysis.json"
        detailed_analysis = analyze_pdf_file(pdf_file, detailed=True)
        with open(output_file, 'w') as f:
            json.dump(detailed_analysis, f, indent=2, default=str)
        print(f"\nDetailed analysis saved to: {output_file}")

        # Extract and show sample of raw PDF commands (first page only)
        print(f"\nRaw PDF Commands (first 2000 chars of page 0):")
        print("-" * 80)
        raw_commands = extract_raw_commands(pdf_file, 0)
        print(raw_commands[:2000])

        # Save full raw commands
        raw_output = Path(pdf_file).stem + "_raw_commands.txt"
        with open(raw_output, 'w') as f:
            f.write(raw_commands)
        print(f"\nFull raw commands saved to: {raw_output}")
