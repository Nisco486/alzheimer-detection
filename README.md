# 🧠 Alzheimer's Disease Detection System

An advanced deep learning system for detecting Alzheimer's disease stages from MRI brain scans using a hybrid CNN + Vision Transformer architecture.

## 🎯 Overview

This project implements a state-of-the-art deep learning model that classifies MRI brain scans into four Alzheimer's disease stages:
- **Non-Demented**: No signs of dementia
- **Very Mild Demented**: Early-stage cognitive decline
- **Mild Demented**: Moderate cognitive impairment
- **Moderate Demented**: Advanced cognitive decline

The system achieves **≥92% accuracy** using a hybrid architecture that combines:
- **EfficientNet-B3** CNN backbone for local feature extraction
- **Vision Transformer (ViT)** with self-attention for global context understanding
- **AI-powered clinical report generation** using LLM integration

## 🏗️ Architecture

```
Input MRI Image (224×224×3)
         ↓
   CNN Backbone (EfficientNet-B3)
         ↓
   Feature Maps (7×7×1536)
         ↓
   Patch Embedding (49 patches)
         ↓
   Vision Transformer (6 layers, 8 heads)
         ↓
   Classification Head
         ↓
   Output (4 classes)
```

## 📊 Dataset

- **Source**: OASIS (Open Access Series of Imaging Studies) - Alzheimer's MRI 4-Classes Dataset
- **Classes**: 4 (Non-Demented, Very Mild, Mild, Moderate)
- **Split**: 70% train, 15% validation, 15% test
- **Preprocessing**: Normalization, resizing to 224×224, data augmentation

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Nisco486/alzheimer-detection.git
cd alzheimer-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
GROQ_BASE_URL=https://api.groq.com/openai/v1
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

### 4. Download Dataset
```bash
python scripts/download_data.py
```

### 5. Run Streamlit App
```bash
streamlit run app/streamlit_app.py
```
The app will open at `http://localhost:8501`

## 📁 Project Structure

```
alzheimer-detection/
│
├── app/
│   ├── streamlit_app.py         # Streamlit web interface
│   ├── inference.py             # Model inference logic
│   ├── agent.py                 # AI clinical report generator
│   └── grad_cam.py              # Grad-CAM visualization
│
├── utils/
│   ├── data_loader.py           # Dataset handling
│   ├── preprocessing.py         # Image preprocessing
│   ├── augmentation.py          # Data augmentation
│   └── metrics.py               # Evaluation metrics
│
├── scripts/
│   └── download_data.py         # Dataset download script
│
├── config.yaml                   # Model configuration
├── train_cnn_model.py           # Training script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🎨 Features

### Web Interface (Streamlit)
- **Image Upload**: Drag-and-drop MRI scan upload
- **Real-time Prediction**: Instant AI-powered diagnosis
- **Confidence Scores**: Probability distribution visualization
- **Grad-CAM Heatmaps**: Visual explanation of model decisions
- **AI Clinical Reports**: Automated clinical report generation with:
  - Patient summary
  - Detailed findings
  - Risk assessment
  - Recommended actions
  - Questions for specialists

### Model Features
- Hybrid CNN + Vision Transformer architecture
- Transfer learning with pre-trained EfficientNet-B3
- Advanced data augmentation (elastic transforms, grid distortion)
- Mixed precision training (AMP)
- Class weighting for imbalanced data
- Early stopping with patience
- Automatic checkpointing

## 📈 Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ≥92%   |
| Precision | ~0.91  |
| Recall    | ~0.90  |
| F1-Score  | ~0.90  |

## 🛠️ Technical Stack

- **Deep Learning**: PyTorch, timm, einops
- **Data Processing**: NumPy, Pandas, Albumentations
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web Interface**: Streamlit
- **AI Integration**: OpenAI API (via Groq), pydantic-ai
- **Development**: Python 3.8+

## 💡 Usage Example

### Inference with Python
```python
from app.inference import predict_image, load_model_from_checkpoint
from PIL import Image

# Load model
model, device = load_model_from_checkpoint('models/best_model.pth', config)

# Predict
image = Image.open('mri_scan.jpg')
prediction, probabilities = predict_image(image, model, device, config)

print(f"Diagnosis: {config['data']['class_names'][prediction]}")
print(f"Confidence: {probabilities[prediction]*100:.1f}%")
```

## 🚨 Medical Disclaimer

**IMPORTANT**: This tool is for research and educational purposes only. It should NOT be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice and treatment decisions.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- OASIS dataset providers
- Google Colab for GPU resources
- PyTorch and Hugging Face communities
- Groq for LLM API access

## 🔗 References

1. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021
2. Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019
3. Marcus et al., "Open Access Series of Imaging Studies (OASIS)", J Cogn Neurosci 2007

---

⭐ **Star this repository if you find it helpful!**