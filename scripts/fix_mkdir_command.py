import json
import os

notebook_path = 'notebooks/train_model.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if 'source' in cell:
        new_source = []
        for line in cell['source']:
            if "!mkdir -p" in line:
                # Replace !mkdir with cross-platform Python code
                line = "import os\n"
                new_source.append(line)
                new_source.append("os.makedirs('data/raw', exist_ok=True)\n")
                new_source.append("os.makedirs('checkpoints', exist_ok=True)\n")
                new_source.append("os.makedirs('logs', exist_ok=True)\n")
                continue # Skip the original mkdir line
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Replaced mkdir command with os.makedirs.")
