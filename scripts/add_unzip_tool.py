import json
import os

notebook_path = 'notebooks/train_model.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Create a clear diagnostic cell for the "Local vs Remote" confusion
diag_cell = [
    "# 🔍 DATA DIAGNOSTIC TOOL\n",
    "import os\n",
    "\n",
    "print(\"Checking for data on the Remote Server...\")\n",
    "\n",
    "# 1. Count actual images\n",
    "image_count = 0\n",
    "found_files = []\n",
    "\n",
    "# Walk through current directory and /content to find ANY jpg/png\n",
    "search_dirs = ['.', '/content']\n",
    "for search_dir in search_dirs:\n",
    "    if os.path.exists(search_dir):\n",
    "        for root, dirs, files in os.walk(search_dir):\n",
    "            for file in files:\n",
    "                if file.lower().endswith(('.jpg', '.jpeg', '.png')):\n",
    "                    image_count += 1\n",
    "                    if len(found_files) < 3:\n",
    "                        found_files.append(os.path.join(root, file))\n",
    "\n",
    "print(f\"\\n📊 Total Images Found on Server: {image_count}\")\n",
    "\n",
    "if image_count == 0:\n",
    "    print(\"❌ PROBLEM DETECTED: No images found on this server.\")\n",
    "    print(\"\\n💡 EXPLANATION:\")\n",
    "    print(\"   You are running this notebook on a Remote GPU Server (Linux).\")\n",
    "    print(\"   But your files are likely on your Local Computer (Windows).\")\n",
    "    print(\"   The server cannot see your local files automatically.\")\n",
    "    print(\"\\n🚀 SOLUTION:\")\n",
    "    print(\"   1. Zip your 'raw' folder on your computer.\")\n",
    "    print(\"   2. Drag and drop the 'raw.zip' file into the file list on the left (in VS Code or Colab interface).\")\n",
    "    print(\"   3. Run the next cell to unzip it.\")\n",
    "    \n",
    "    data_root = None\n",
    "else:\n",
    "    print(\"✅ Images found! attempting to locate root folder...\")\n",
    "    # Attempt to derive data_root from the first found file\n",
    "    # e.g. /content/data/raw/Non Demented/img1.jpg -> /content/data/raw\n",
    "    first_img = found_files[0]\n",
    "    # Go up two levels\n",
    "    parent = os.path.dirname(first_img) # Non Demented\n",
    "    grandparent = os.path.dirname(parent) # raw\n",
    "    data_root = grandparent\n",
    "    print(f\"📂 Derived Data Root: {data_root}\")\n"
]

# Unzip helper cell
unzip_cell = [
    "# 🛠️ UNZIP TOOL (Run this after uploading raw.zip)\n",
    "import zipfile\n",
    "import os\n",
    "\n",
    "zip_path = 'raw.zip'\n",
    "extract_to = 'data/raw'\n",
    "\n",
    "if os.path.exists(zip_path):\n",
    "    print(f\"📦 Found {zip_path}! Extracting...\")\n",
    "    with zipfile.ZipFile(zip_path, 'r') as zip_ref:\n",
    "        zip_ref.extractall(extract_to)\n",
    "    print(f\"✅ Extracted to {extract_to}\")\n",
    "    data_root = extract_to\n",
    "    if os.path.exists(os.path.join(extract_to, 'Data')):\n",
    "        data_root = os.path.join(extract_to, 'Data')\n",
    "        print(f\"   (Adjusted root to inner 'Data' folder: {data_root})\")\n",
    "else:\n",
    "    print(\"⚠️ 'raw.zip' not found. If you have data already, compare the 'Derived Data Root' above.\")\n"
]

for cell in nb['cells']:
    if 'source' in cell:
        source_text = "".join(cell['source'])
        if "Searching for dataset" in source_text or "DATA DIAGNOSTIC" in source_text:
            cell['source'] = diag_cell

# Append the unzip cell after the diag cell if it doesn't exist
has_unzip = False
for cell in nb['cells']:
    if "UNZIP TOOL" in "".join(cell.get('source', [])):
        has_unzip = True

if not has_unzip:
    # Find index of diag cell and insert after
    for i, cell in enumerate(nb['cells']):
        if "DATA DIAGNOSTIC" in "".join(cell.get('source', [])):
             nb['cells'].insert(i + 1, {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": unzip_cell
             })
             break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Added diagnostic and unzip tools to notebook.")
