import json
import os

notebook_path = 'notebooks/train_model.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Create a robust search-only cell (No Drive Mount)
search_cell = [
    "# Auto-Discovery of Dataset (No Drive Mount)\n",
    "import os\n",
    "\n",
    "print(\"🔍 Searching for dataset...\")\n",
    "\n",
    "required_folders = {'Non Demented', 'Mild Dementia'}\n",
    "found_data_root = None\n",
    "\n",
    "# 1. Define search zones based on environment\n",
    "search_roots = ['/content', '.']\n",
    "\n",
    "# 2. Walk and find\n",
    "for search_root in search_roots:\n",
    "    if not os.path.exists(search_root):\n",
    "        continue\n",
    "        \n",
    "    for root, dirs, files in os.walk(search_root):\n",
    "        # Check if current dir has the required class folders\n",
    "        if required_folders.issubset(set(dirs)):\n",
    "            found_data_root = root\n",
    "            break\n",
    "    if found_data_root:\n",
    "        break\n",
    "\n",
    "if found_data_root:\n",
    "    print(f\"✅ Dataset FOUND at: {found_data_root}\")\n",
    "    data_root = found_data_root\n",
    "else:\n",
    "    print(\"❌ Dataset NOT FOUND.\")\n",
    "    print(\"   Please ensure you have uploaded the 'Data' folder containing 'Mild Dementia' etc.\")\n",
    "    print(\"   If on Colab, drag-and-drop the folder into the Files sidebar.\")\n",
    "    # Fallback to prevent crash in config, though it will fail later if not fixed\n",
    "    data_root = 'DATA_NOT_FOUND'\n"
]

for cell in nb['cells']:
    if 'source' in cell:
        source_text = "".join(cell['source'])
        # Replace any cell that initiates mounting or deep searching with this new passive search
        if "Mounting Google Drive" in source_text or "Deep Search" in source_text or "Intelligent Data Loader" in source_text:
            cell['source'] = search_cell
        # Ensure config uses the variable
        if "config = {" in source_text:
            new_source = []
            for line in cell['source']:
                if "'root_dir':" in line:
                    line = "        'root_dir': data_root,\n"
                new_source.append(line)
            cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Replaced Drive mounting with auto-discovery search.")
