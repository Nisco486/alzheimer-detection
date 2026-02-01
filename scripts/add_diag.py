import json
import os

notebook_path = 'notebooks/train_model.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Create a diagnostic cell
diag_source = [
    "import os, sys, platform\n",
    "print(f\"OS: {os.name}\")\n",
    "print(f\"Platform: {sys.platform}\")\n",
    "print(f\"Python Version: {sys.version}\")\n",
    "print(f\"Current Working Directory: {os.getcwd()}\")\n",
    "print(\"Directory Listing of current dir:\")\n",
    "print(os.listdir('.'))\n",
    "\n",
    "print(\"\\n--- Drive/Mount Checks ---\")\n",
    "if os.path.exists('/content'):\n",
    "    print(\"✅ Found /content (Likely Colab)\")\n",
    "else:\n",
    "    print(\"❌ /content not found\")\n",
    "\n",
    "if os.path.exists('/mnt/c'):\n",
    "    print(\"✅ Found /mnt/c (Likely WSL)\")\n",
    "else:\n",
    "    print(\"❌ /mnt/c not found\")\n"
]

# Insert this as the very first cell
nb['cells'].insert(0, {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": diag_source
})

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Added diagnostic cell to top of notebook.")
