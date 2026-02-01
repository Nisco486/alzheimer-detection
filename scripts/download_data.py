import os
import shutil
import zipfile
import subprocess
import sys
import json
from dotenv import load_dotenv

# Load env from root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# 1. Setup Kaggle Config
home = os.path.expanduser("~")
kaggle_dir = os.path.join(home, ".kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

dest_kaggle = os.path.join(kaggle_dir, "kaggle.json")

# Get from env
kaggle_user = os.getenv("KAGGLE_USERNAME")
kaggle_key = os.getenv("KAGGLE_KEY")

if kaggle_user and kaggle_key:
    with open(dest_kaggle, "w") as f:
        json.dump({"username": kaggle_user, "key": kaggle_key}, f)
    print(f"✅ Created/Updated kaggle.json in {kaggle_dir} from environment variables.")
else:
    print(f"⚠️ Kaggle credentials not found in environment variables. Assuming it's already in {kaggle_dir}")

# 2. Download Dataset
dataset = "marcopinamonti/alzheimer-mri-4-classes-dataset"
output_zip = "data/raw/alzheimer-mri.zip"
os.makedirs("data/raw", exist_ok=True)

print(f"⬇️ Downloading {dataset}...")
try:
    # Use explicit python executable to run kaggle module to avoid path issues
    python_exe = sys.executable
    subprocess.run([python_exe, "-m", "kaggle", "datasets", "download", "-d", dataset, "-p", "data/raw", "--force"], check=True)
    print("✅ Download Complete.")
except subprocess.CalledProcessError as e:
    print(f"❌ Error downloading: {e}")
    # Fallback: try installing if module not found (though we just installed it)
    try:
         subprocess.run([python_exe, "-m", "pip", "install", "kaggle"], check=True)
         subprocess.run([python_exe, "-m", "kaggle", "datasets", "download", "-d", dataset, "-p", "data/raw", "--force"], check=True)
    except Exception as e2:
         print(f"❌ Critical failure: {e2}")

# 3. Unzip
target_zip = "data/raw/alzheimer-mri-4-classes-dataset.zip" # Kaggle default name
if os.path.exists(target_zip):
    print("📦 Unzipping...")
    with zipfile.ZipFile(target_zip, 'r') as zip_ref:
        zip_ref.extractall("data/raw")
    print("✅ Unzip Complete. Data is in data/raw/")
else:
    print(f"⚠️ Could not find downloaded zip at {target_zip}")
