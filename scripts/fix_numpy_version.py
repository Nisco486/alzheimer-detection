import json
import os

notebook_path = 'notebooks/train_model.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if 'source' in cell:
        new_source = []
        for line in cell['source']:
            if "!pip install" in line and "numpy<2.0" not in line:
                # Add numpy constraint to existing pip command
                line = line.replace("PyYAML==6.0.2", 'PyYAML==6.0.2 "numpy<2.0"')
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Added numpy<2.0 constraint to notebook pip command.")
