import json
import os

notebook_path = 'notebooks/train_model.ipynb'
absolute_data_path = 'C:/Users/nisha/OneDrive/Documents/alzheimer-detection/data/raw'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update the "Check/Mount" cell
for cell in nb['cells']:
    if 'source' in cell:
        source_text = "".join(cell['source'])
        if "from google.colab import drive" in source_text or "Local data found" in source_text:
            cell['source'] = [
                "# Using Absolute Path for Local Data to avoid CWD issues\n",
                "import os\n",
                f"data_root = '{absolute_data_path}'\n",
                "\n",
                "print(f\"📂 Using Data Directory: {data_root}\")\n",
                "if os.path.exists(data_root):\n",
                "    print(f\"✅ Data found! Contents: {os.listdir(data_root)}\")\n",
                "else:\n",
                "    print(\"❌ Data path not found!\")\n"
            ]
            print("Updated Data Check cell.")

# 2. Update the Config cell
for cell in nb['cells']:
    if 'source' in cell:
        source_text = "".join(cell['source'])
        if "config = {" in source_text:
            new_source = []
            for line in cell['source']:
                if "'root_dir':" in line:
                    # Inject variable or hardcoded path
                    line = f"        'root_dir': '{absolute_data_path}',   # ✅ absolute path\n"
                new_source.append(line)
            cell['source'] = new_source
            print("Updated Config cell.")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated with absolute paths.")
