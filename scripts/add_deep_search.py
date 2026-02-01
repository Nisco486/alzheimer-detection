import json
import os

notebook_path = 'notebooks/train_model.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Create a Deep Search/Repair Cell for Colab
search_cell_source = [
    "import os\n",
    "\n",
    "print(\"🔍 Deep Search for Dataset in Colab...\")\n",
    "top_level = '/content'\n",
    "found_path = None\n",
    "\n",
    "target_folders = {'Non Demented', 'Mild Dementia'}\n",
    "\n",
    "# Walk through the entire /content directory to find the class folders\n",
    "for root, dirs, files in os.walk(top_level):\n",
    "    # Check if this directory contains our target class folders\n",
    "    if target_folders.issubset(set(dirs)):\n",
    "        found_path = root\n",
    "        print(f\"✅ FOUND DATASET AT: {found_path}\")\n",
    "        break\n",
    "\n",
    "if found_path:\n",
    "    data_root = found_path\n",
    "    print(f\"   Contents: {os.listdir(data_root)}\")\n",
    "else:\n",
    "    print(\"❌ Dataset NOT FOUND anywhere in /content\")\n",
    "    print(\"📂 Listing /content/data anyway:\")\n",
    "    if os.path.exists('/content/data'):\n",
    "        for root, dirs, files in os.walk('/content/data'):\n",
    "              print(f\"  {root}/ -> {dirs}\")\n",
    "    else:\n",
    "        print(\"  /content/data does not exist.\")\n"
]

for cell in nb['cells']:
    if 'source' in cell:
        source_text = "".join(cell['source'])
        if "Deep Search" in source_text or "Could not find dataset" in source_text or "Found Dataset at" in source_text:
            cell['source'] = search_cell_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Added deep search cell to notebook.")
