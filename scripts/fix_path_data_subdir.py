import json
import os

notebook_path = 'notebooks/train_model.ipynb'
# The correct path actually INCLUDES /Data based on latest list_dir check
correct_path = 'C:/Users/nisha/OneDrive/Documents/alzheimer-detection/data/raw/Data'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if 'source' in cell:
        new_source = []
        for line in cell['source']:
            # Fix data_root assignment
            if "data_root =" in line:
                line = f"    data_root = '{correct_path}'\n"
            
            # Fix config root_dir assignment
            if "'root_dir':" in line:
                line = f"        'root_dir': '{correct_path}',   # ✅ absolute path with Data subdir\n"
                
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook path corrected to include /Data subdirectory.")
