import json
import os

notebook_path = 'notebooks/train_model.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if 'source' in cell:
        new_source = []
        for line in cell['source']:
            if "!pip install" in line:
                # Add commonly missing data science libs to the install command
                if "matplotlib" not in line:
                    line = line.strip() + " matplotlib seaborn scikit-learn pandas\n"
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated notebook install command to include matplotlib, seaborn, etc.")
