import json
import os

notebook_path = 'notebooks/train_model.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Remove the diagnostic cell (it was the first one)
if 'diagnostic' in str(nb['cells'][0].get('source')):
    del nb['cells'][0]
# Or just check if it imports sys and platform
if len(nb['cells']) > 0 and 'import os, sys, platform' in "".join(nb['cells'][0]['source']):
    print("Removing diagnostic cell.")
    del nb['cells'][0]

# 2. Update the Path Selection Cell
colab_path_source = [
    "# Configured for Colab Environment\n",
    "import os\n",
    "\n",
    "# Based on your current environment (Colab), we check these paths:\n",
    "possible_paths = [\n",
    "    '/content/data/raw/Data',  # Most likely based on your upload\n",
    "    '/content/data/raw',       # Alternative structure\n",
    "    '/content/data'            # Simple structure\n",
    "]\n",
    "\n",
    "data_root = None\n",
    "for p in possible_paths:\n",
    "    if os.path.exists(p):\n",
    "        # Check if it actually contains the class folders\n",
    "        contents = os.listdir(p)\n",
    "        if 'Non Demented' in contents or 'Mild Dementia' in contents:\n",
    "            data_root = p\n",
    "            break\n",
    "\n",
    "if data_root:\n",
    "    print(f\"✅ Found Dataset at: {data_root}\")\n",
    "    print(f\"   Contents: {os.listdir(data_root)}\")\n",
    "else:\n",
    "    print(\"❌ Could not find dataset folders (Non Demented, etc) in /content/data\")\n",
    "    print(\"   Please verify you uploaded the 'raw' folder correctly.\")\n",
    "    # Fallback to just /content/data/raw to show error clearly\n",
    "    data_root = '/content/data/raw'\n"
]

for cell in nb['cells']:
    if 'source' in cell:
        source_text = "".join(cell['source'])
        # Find the cell trying to set data_root or checking paths
        if "data_root =" in source_text and ("possible_paths" in source_text or "Robust Data Path" in source_text or "Using Absolute Path" in source_text):
            cell['source'] = colab_path_source
            
        # Update config to use data_root variable
        if "config = {" in source_text:
             new_source = []
             for line in cell['source']:
                 if "'root_dir':" in line:
                     line = "        'root_dir': data_root,\n"
                 new_source.append(line)
             cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated for Colab environment.")
