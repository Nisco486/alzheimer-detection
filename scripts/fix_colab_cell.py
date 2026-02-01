import json
import os

notebook_path = 'notebooks/train_model.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and Replace the Colab Cell
for cell in nb['cells']:
    if 'source' in cell:
        source_text = "".join(cell['source'])
        if "from google.colab import drive" in source_text:
            cell['source'] = [
                "# Mount Google Drive - SKIPPING for Local Run\n",
                "# from google.colab import drive\n",
                "# drive.mount('/content/drive')\n",
                "\n",
                "# For local run, we check if data exists where we expect it\n",
                "import os\n",
                "if os.path.exists('data/raw'):\n",
                "    print(\"✅ Local data found at data/raw\")\n",
                "    print(f\"Contents: {os.listdir('data/raw')}\")\n",
                "elif os.path.exists('../data/raw'):\n",
                "    print(\"✅ Local data found at ../data/raw\")\n",
                "    print(f\"Contents: {os.listdir('../data/raw')}\")\n",
                "else:\n",
                "    print(\"❌ Data not found. Please check your data directory.\")\n"
            ]
            print("Replaced Google Drive cell with local check.")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
