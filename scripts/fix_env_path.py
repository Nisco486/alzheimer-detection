import json
import os

notebook_path = 'notebooks/train_model.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Create a robust, environment-aware data loading cell
robust_cell_source = [
    "# Robust Data Path Setup (Works for Local and Colab)\n",
    "import os\n",
    "import sys\n",
    "\n",
    "def get_data_path():\n",
    "    # 1. Check if running in Google Colab\n",
    "    if 'google.colab' in sys.modules:\n",
    "        from google.colab import drive\n",
    "        print(\"☁️ Detected Google Colab Environment\")\n",
    "        if not os.path.exists('/content/drive'):\n",
    "            drive.mount('/content/drive')\n",
    "        \n",
    "        # Check common Colab paths (adjust these if your Drive structure is different)\n",
    "        colab_paths = [\n",
    "            '/content/drive/MyDrive/alzheimer-detection/data/raw/Data',\n",
    "            '/content/drive/MyDrive/alzheimer-dataset/data/raw/Data',\n",
    "            '/content/data/raw/Data'\n",
    "        ]\n",
    "        for p in colab_paths:\n",
    "            if os.path.exists(p):\n",
    "                return p\n",
    "                \n",
    "        print(\"⚠️ Data not found in common Drive locations. Please check your Drive structure.\")\n",
    "        return None\n",
    "\n",
    "    # 2. Local Environment Checks\n",
    "    else:\n",
    "        print(\"🖥️ Detected Local Environment\")\n",
    "        local_paths = [\n",
    "            'C:/Users/nisha/OneDrive/Documents/alzheimer-detection/data/raw/Data',\n",
    "            'data/raw/Data',\n",
    "            '../data/raw/Data',\n",
    "            '../../data/raw/Data'\n",
    "        ]\n",
    "        for p in local_paths:\n",
    "            if os.path.exists(p):\n",
    "                return p\n",
    "    \n",
    "    return None\n",
    "\n",
    "# Set the path\n",
    "data_root = get_data_path()\n",
    "\n",
    "if data_root:\n",
    "    print(f\"✅ Data Directory Found: {data_root}\")\n",
    "    print(f\"Contents: {os.listdir(data_root)}\")\n",
    "else:\n",
    "    print(\"❌ Data path not found anywhere!\")\n",
    "    if 'google.colab' in sys.modules:\n",
    "        print(\"👉 If you are on Colab, make sure you have uploaded the data to Drive and updated the path in the code.\")\n"
]

# Update the notebook
for cell in nb['cells']:
    if 'source' in cell:
        source_text = "".join(cell['source'])
        # Replace the hardcoded absolute path cell we made earlier
        if "C:/Users/nisha" in source_text and "Using Data Directory" in source_text:
            cell['source'] = robust_cell_source
            
        # Update config to use the dynamic variable 'data_root'
        if "config = {" in source_text:
            new_source = []
            for line in cell['source']:
                if "'root_dir':" in line:
                    line = "        'root_dir': data_root,   # ✅ Uses dynamically found path\n"
                new_source.append(line)
            cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated notebook with robust environment-aware data path logic.")
