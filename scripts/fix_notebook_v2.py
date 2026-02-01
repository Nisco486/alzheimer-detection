import json
import os

notebook_path = 'notebooks/train_model.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix PyYAML dependency
for cell in nb['cells']:
    if 'source' in cell:
        new_source = []
        for line in cell['source']:
            if 'pip install' in line and 'PyYAML' in line:
                line = line.replace('PyYAML==6.0.1', 'PyYAML==6.0.2')
            new_source.append(line)
        cell['source'] = new_source

# Fix Config Path and Add CWD check
found_config = False
for cell in nb['cells']:
    if 'source' in cell:
        source_text = "".join(cell['source'])
        
        # Add CWD fix to imports cell
        if "import os" in source_text and "warnings.filterwarnings" in source_text:
             if "os.chdir" not in source_text:
                cell['source'].insert(1, "if os.path.basename(os.getcwd()) == 'notebooks':\n    os.chdir('..')\nprint(f'Current Working Directory: {os.getcwd()}')\n")

        # Fix config root_dir
        if "config = {" in source_text:
            new_source = []
            for line in cell['source']:
                if "'root_dir':" in line:
                    # Set it to data/raw which works if we are at project root (ensured by CWD fix)
                    line = "        'root_dir': 'data/raw',   # ✅ correct level\n"
                new_source.append(line)
            cell['source'] = new_source
            found_config = True

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
