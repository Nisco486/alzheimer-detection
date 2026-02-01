import json
import os

notebook_path = 'notebooks/train_model.ipynb'
correct_path = 'C:/Users/nisha/OneDrive/Documents/alzheimer-detection/data/raw'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix every occurrence of the path
for cell in nb['cells']:
    if 'source' in cell:
        new_source = []
        for line in cell['source']:
            # Fix data_root assignment
            if "data_root =" in line and "/Data'" in line:
                line = line.replace("/Data'", "'")
            
            # Fix config root_dir assignment if it has /Data
            if "'root_dir':" in line and "/Data'" in line:
                line = line.replace("/Data'", "'")
                
            # Ensure PyYAML is 6.0.2
            if "PyYAML==6.0.1" in line:
                line = line.replace("PyYAML==6.0.1", "PyYAML==6.0.2")

            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook path and dependencies fixed.")
