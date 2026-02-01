import json
import os

notebook_path = 'notebooks/train_model.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Improved Data Loading Cell with Drive Mount
loader_source = [
    "# Intelligent Data Loader (Works for Colab & Local)\n",
    "import os\n",
    "import sys\n",
    "\n",
    "data_root = None\n",
    "\n",
    "def check_path(path):\n",
    "    if os.path.exists(path):\n",
    "        contents = os.listdir(path)\n",
    "        # Check for expected class folders\n",
    "        if 'Non Demented' in contents or 'Mild Dementia' in contents:\n",
    "            return True\n",
    "    return False\n",
    "\n",
    "# 1. Check Local Environment (Colab temporary or Local Machine)\n",
    "possible_paths = [\n",
    "    '/content/data/raw/Data',\n",
    "    '/content/data/raw',\n",
    "    'C:/Users/nisha/OneDrive/Documents/alzheimer-detection/data/raw/Data',\n",
    "    'data/raw/Data'\n",
    "]\n",
    "\n",
    "for p in possible_paths:\n",
    "    if check_path(p):\n",
    "        data_root = p\n",
    "        print(f\"✅ Found dataset locally at: {data_root}\")\n",
    "        break\n",
    "\n",
    "# 2. If NOT found and we are in Colab, Mount Drive\n",
    "if data_root is None and 'google.colab' in sys.modules:\n",
    "    print(\"⚠️ Data not found locally. Mounting Google Drive...\")\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive')\n",
    "    \n",
    "    # Search Drive\n",
    "    drive_paths = [\n",
    "        '/content/drive/MyDrive/alzheimer-detection/data/raw/Data',\n",
    "        '/content/drive/MyDrive/alzheimer-detection/data/raw',\n",
    "        '/content/drive/MyDrive/alzheimer-dataset/data/raw',\n",
    "        '/content/drive/MyDrive/Data/raw'\n",
    "    ]\n",
    "    \n",
    "    for p in drive_paths:\n",
    "        if check_path(p):\n",
    "            data_root = p\n",
    "            print(f\"✅ Found dataset in Drive at: {data_root}\")\n",
    "            break\n",
    "\n",
    "if data_root:\n",
    "    print(f\"📂 Data Root set to: {data_root}\")\n",
    "else:\n",
    "    print(\"❌ CRITICAL: Dataset not found!\")\n",
    "    print(\"   Please upload your 'raw' folder to Google Drive at 'MyDrive/alzheimer-detection/data/raw'\")\n",
    "    print(\"   OR upload it directly to Colab files on the left.\")\n"
]

for cell in nb['cells']:
    if 'source' in cell:
        source_text = "".join(cell['source'])
        # Replace the previous search cell or the config cell
        if "Deep Search" in source_text or "Robust Data Path" in source_text:
            cell['source'] = loader_source
        
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

print("Restored Google Drive mounting logic to notebook.")
