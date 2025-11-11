#!/usr/bin/env python3
import fitz
import json

pdf_path = "/mnt/d/mg_ai_research/workspace/cadAI/pdf2cad2/pdf_files/60355K178_Ball Bearing.pdf"
doc = fitz.open(pdf_path)
page = doc[0]
paths = page.get_drawings()

print(f"Total paths: {len(paths)}")
print(f"\nFirst path structure:")
print(f"Type: {type(paths[0])}")
print(f"Keys: {paths[0].keys() if isinstance(paths[0], dict) else 'Not a dict'}")

if len(paths) > 0:
    path = paths[0]
    print(f"\nFirst path content:")
    for key, value in path.items():
        print(f"  {key}: {type(value)} = {value if key != 'items' else f'[{len(value)} items]'}")

    if 'items' in path and len(path['items']) > 0:
        print(f"\nFirst item in items:")
        first_item = path['items'][0]
        print(f"  Type: {type(first_item)}")
        print(f"  Content: {first_item}")
        print(f"  Length: {len(first_item)}")
        for i, elem in enumerate(first_item):
            print(f"    [{i}]: {type(elem)} = {elem}")
