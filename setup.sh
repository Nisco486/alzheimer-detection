#!/bin/bash

# Setup script for Alzheimer's Detection Project
# Run this after cloning the repository

echo "🧠 Setting up Alzheimer's Detection Project..."
echo "================================================"

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/splits
mkdir -p checkpoints
mkdir -p logs
mkdir -p models
mkdir -p utils
mkdir -p app
mkdir -p notebooks

# Create __init__.py files
echo "📝 Creating __init__.py files..."
touch models/__init__.py
touch utils/__init__.py
touch app/__init__.py

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Download dataset (optional)
echo ""
echo "📊 Dataset Setup"
echo "================"
echo "To download the OASIS dataset from Kaggle:"
echo "1. Install Kaggle CLI: pip install kaggle"
echo "2. Set up Kaggle API credentials (~/.kaggle/kaggle.json)"
echo "3. Run: kaggle datasets download -d kirollosashraf/oasis-alzheimers-detection"
echo "4. Unzip to data/raw/"
echo ""
read -p "Do you want to download the dataset now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Downloading dataset..."
    kaggle datasets download -d kirollosashraf/oasis-alzheimers-detection
    echo "Unzipping dataset..."
    unzip oasis-alzheimers-detection.zip -d data/raw/
    rm oasis-alzheimers-detection.zip
    echo "✅ Dataset downloaded and extracted!"
fi

# Create sample config if doesn't exist
if [ ! -f config.yaml ]; then
    echo "⚙️ config.yaml already exists or will be created from artifact"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📚 Next Steps:"
echo "=============="
echo "1. If you haven't downloaded the dataset, do so manually"
echo "2. Open notebooks/train_model.ipynb in VS Code"
echo "3. Connect to Google Colab T4 GPU"
echo "4. Train the model (2-3 hours)"
echo "5. Download trained model from Google Drive"
echo "6. Place it in checkpoints/best_model.pth"
echo "7. Run: streamlit run app/streamlit_app.py"
echo ""
echo "🚀 Happy training!"