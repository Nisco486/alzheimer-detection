import json
import os

notebook_path = 'notebooks/train_model.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Smarter Unzip Cell
smart_unzip_source = [
    "# 🛠️ SMART UNZIP TOOL\n",
    "import zipfile\n",
    "import os\n",
    "\n",
    "# Possible locations where raw.zip might be suitable\n",
    "potential_zips = [\n",
    "    'raw.zip', \n",
    "    'notebooks/raw.zip',\n",
    "    '../raw.zip',\n",
    "    '/content/raw.zip',\n",
    "    '/content/notebooks/raw.zip'\n",
    "]\n",
    "\n",
    "found_zip = None\n",
    "print(\"🔍 Looking for raw.zip...\")\n",
    "for p in potential_zips:\n",
    "    if os.path.exists(p):\n",
    "        found_zip = p\n",
    "        print(f\"✅ Found zip file at: {found_zip}\")\n",
    "        break\n",
    "\n",
    "if found_zip:\n",
    "    # Determine extraction path. If we are in 'notebooks', extract to '../data/raw' or 'data/raw' depending on structure\n",
    "    # Safest bet: Extract to a known absolute data dir or relative 'data/raw'\n",
    "    extract_to = 'data/raw'\n",
    "    \n",
    "    # If the zip is in 'notebooks/', we might want to extract to '../data/raw' if we are in project root\n",
    "    # BUT, let's stick to current directory 'data/raw' to be safe and use that as root.\n",
    "    \n",
    "    print(f\"📦 Extracting to '{extract_to}'...\")\n",
    "    os.makedirs(extract_to, exist_ok=True)\n",
    "    \n",
    "    with zipfile.ZipFile(found_zip, 'r') as zip_ref:\n",
    "        zip_ref.extractall(extract_to)\n",
    "        \n",
    "    print(\"✅ Extraction Complete.\")\n",
    "    \n",
    "    # Verify and Set Root\n",
    "    if os.path.exists(os.path.join(extract_to, 'Data')):\n",
    "        data_root = os.path.join(extract_to, 'Data')\n",
    "        print(f\"📂 Data Root set to inner folder: {data_root}\")\n",
    "    elif os.path.exists(os.path.join(extract_to, 'Non Demented')):\n",
    "        data_root = extract_to\n",
    "        print(f\"📂 Data Root set to: {data_root}\")\n",
    "    elif os.path.exists(os.path.join(extract_to, 'raw', 'Data')):\n",
    "        data_root = os.path.join(extract_to, 'raw', 'Data')\n",
    "        print(f\"📂 Data Root set to nested: {data_root}\")\n",
    "    else:\n",
    "        # Just list what we have\n",
    "        print(f\"⚠️ Extracted, but check structure. Contents of {extract_to}: {os.listdir(extract_to)}\")\n",
    "        data_root = extract_to\n",
    "else:\n",
    "    print(\"❌ 'raw.zip' NOT FOUND.\")\n",
    "    print(f\"   Checked CWD: {os.getcwd()}\")\n"
]

for cell in nb['cells']:
    if 'source' in cell:
        source_text = "".join(cell['source'])
        if "UNZIP TOOL" in source_text or "smart_unzip" in source_text:
            cell['source'] = smart_unzip_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated unzip tool to search multiple paths.")
