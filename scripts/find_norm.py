import json

with open('alzheimer.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    source = ''.join(cell.get('source', []))
    if 'Normalize' in source or 'transforms' in source or 'preprocessing' in source:
        print(f"--- Cell ---")
        print(source)
