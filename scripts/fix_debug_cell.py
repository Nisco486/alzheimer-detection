import json
import os

notebook_path = 'notebooks/train_model.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update the local data verification cell to be more robust and debug friendly
for cell in nb['cells']:
    if 'source' in cell:
        source_text = "".join(cell['source'])
        if "✅ Local data found" in source_text or "SKIP" in source_text:
             cell['source'] = [
                "# Mount Google Drive - SKIPPING for Local Run\n",
                "# from google.colab import drive\n",
                "# drive.mount('/content/drive')\n",
                "\n",
                "# For local run, we check if data exists where we expect it\n",
                "import os\n",
                "\n",
                "# Debug: Print expected path\n",
                "abs_path = os.path.abspath('data/raw')\n",
                "print(f\"Searching in: {abs_path}\")\n",
                "\n",
                "if os.path.exists('data/raw'):\n",
                "    contents = os.listdir('data/raw')\n",
                "    print(f\"✅ Local data found at data/raw\")\n",
                "    print(f\"Contents: {contents}\")\n",
                "    if len(contents) == 0:\n",
                 "        print(\"⚠️ WARNING: The directory exists but appears empty. Check if you are in the correct root directory.\")\n",
                "elif os.path.exists('../data/raw'):\n",
                "    contents = os.listdir('../data/raw')\n",
                "    print(\"✅ Local data found at ../data/raw\")\n",
                "    print(f\"Contents: {contents}\")\n",
                "else:\n",
                "    print(f\"❌ Data NOT found at {abs_path}. Please check your CWD.\")\n",
                "    print(f\"Current CWD: {os.getcwd()}\")\n"
            ]
             print("Updated debugging cell.")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
