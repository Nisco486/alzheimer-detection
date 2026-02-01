from grad_cam import GradCAM
import cv2
import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import yaml
import plotly.graph_objects as go
import asyncio
import nest_asyncio
from inference import load_model_from_checkpoint, predict_image, get_class_name, get_confidence_level, preprocess_image
from agent import generate_clinical_report, PredictionContext

nest_asyncio.apply()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="NeuralScan AI - AD Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top right, #0f172a, #020617);
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
    .header-text {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subheader-text {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    /* Metrics */
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Diagnosis Styles */
    .diagnosis-safe { color: #4ade80; }
    .diagnosis-warning { color: #facc15; }
    .diagnosis-danger { color: #f87171; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: rgba(15, 23, 42, 0.5);
        padding: 0.5rem 1rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8;
        border-bottom-color: #38bdf8;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD CONFIG ---
@st.cache_resource
def get_config():
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        config_path = '../config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- LOAD MODEL ---
# @st.cache_resource
def get_model(architecture='hybrid'):
    config = get_config()
    
    # Map architectures to config paths
    arch_to_config_key = {
        'hybrid': 'hybrid_model',
        'cnn': 'cnn_model',
        'vit': 'vit_model'
    }
    
    model_path = None
    
    # 1. Try to get specific path from config
    if architecture in arch_to_config_key:
        config_key = arch_to_config_key[architecture]
        if config_key in config['paths']:
            model_path = config['paths'][config_key]
            # Handle potential relative path issues
            if not os.path.exists(model_path):
                alt_path = os.path.join('..', model_path)
                if os.path.exists(alt_path):
                    model_path = alt_path
    
    # 2. Fallback to general search if path doesn't exist or isn't in config
    if not model_path or not os.path.exists(model_path):
        possible_paths = [
            f'../models/{architecture}_best.pth',
            '../models/fine_tuned_model.pth',
            '../models/alzheimer_model_final.pth',
            '../models/best_model.pth',
            '../checkpoints/best_model.pth'
        ]
        
        for p in possible_paths:
            if os.path.exists(p):
                model_path = p
                break
            
    # Load model using inference utility
    model, device = load_model_from_checkpoint(model_path, config, architecture)
    return model, device, config, model_path

# --- HELPER FUNCTIONS ---
def plot_probability_chart(probabilities, labels, highlight_idx):
    colors = ['#1e293b'] * len(labels)
    colors[highlight_idx] = '#38bdf8'
    
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=probabilities,
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probabilities],
        textposition='auto',
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=200,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, range=[0, 1])
    )
    return fig

# --- MAIN PAGE ---
def main():
    config = get_config()
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown('## 🧠 NeuralScan AI')
        # st.markdown("Advanced Multi-Model Diagnostics")
        
        st.info("ℹ️ **System Status**\n\n"
                "• Hybrid Model: **Active**\n"
                "• CNN Model: **Available**\n"
                "• ViT Model: **Available**")
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        cam_opacity = st.slider("Grad-CAM Opacity", 0.0, 1.0, 0.4, step=0.1)
        
        st.markdown("---")
        st.markdown("### 📝 About")
        st.caption("This tool uses a Hybrid EfficientNet + Vision Transformer architecture to detect Alzheimer's Disease stages from MRI scans.")
        
        st.markdown("---")
        st.warning("**Disclaimer**: For research use only. Not for clinical diagnosis.")

    # --- MAIN CONTENT ---
    st.markdown('<h1 class="header-text">NeuralScan 🧠</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader-text">AI-Powered Alzheimer\'s Stage Classification from MRI Scans</p>', unsafe_allow_html=True)

    # File Uploader
    uploaded_file = st.file_uploader("Upload Brain MRI Scan", type=["jpg", "jpeg", "png"], help="Upload a coronal slice MRI")

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Tabs for different analysis modes
        tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "📊 Model Comparison", "📋 Clinical Details"])

        # --- TAB 1: MAIN ANALYSIS (HYBRID) ---
        with tab1:
            col_img, col_res = st.columns([1, 1.5])
            
            # Load Primary Model (Hybrid)
            model, device, _, _ = get_model('hybrid')
            
            with st.spinner('Running Hybrid Analysis...'):
                prediction, probabilities = predict_image(image, model, device, config)
                class_name = get_class_name(prediction, config)
                confidence = probabilities[prediction]

            with col_img:
                st.markdown("### Input Scan")
                # Grad-CAM overlay
                input_tensor = preprocess_image(image, config).to(device)
                
                # GradCAM Logic
                try:
                    target_layer = None
                    if hasattr(model, 'cnn_backbone'):
                        # For Hybrid (timm backbone in features_only=True mode)
                        # The last block is typically the best for Grad-CAM
                        try:
                            target_layer = model.cnn_backbone.blocks[-1][-1].conv_pwl
                        except:
                            target_layer = model.cnn_backbone.blocks[-1][-1]
                    elif hasattr(model, 'model'):
                        if hasattr(model.model, 'conv_head'):
                            # For pure CNN (EfficientNet)
                            target_layer = model.model.conv_head
                        elif hasattr(model.model, 'blocks'):
                            # For pure ViT (MLP block)
                            try:
                                target_layer = model.model.blocks[-1].norm2
                            except:
                                target_layer = model.model.norm
                    
                    if target_layer:
                        cam = GradCAM(model, target_layer).generate(input_tensor, prediction)
                        img_np = np.array(image.resize((224, 224)))
                        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                        overlay = cv2.addWeighted(img_np, 1 - cam_opacity, heatmap_rgb, cam_opacity, 0)
                        st.image(overlay, caption="Grad-CAM Attention Map", use_column_width=True)
                    else:
                        st.image(image, caption="Original Image (Grad-CAM unavailable)", use_column_width=True)

                except Exception as e:
                    st.error(f"Grad-CAM Error: {e}")
                    st.image(image, caption="Original Image", use_column_width=True)

            with col_res:
                st.markdown("### Result")
                st.markdown(f'<div class="glass-card"><span class="metric-value" style="color:#38bdf8">{class_name}</span></div>', unsafe_allow_html=True)
                
                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    st.markdown(f'<div class="metric-label">Confidence</div><div class="metric-value">{confidence*100:.1f}%</div>', unsafe_allow_html=True)
                with c_col2:
                    st.markdown(f'<div class="metric-label">Risk Level</div><div class="metric-value">{get_confidence_level(confidence)}</div>', unsafe_allow_html=True)
                
                st.plotly_chart(plot_probability_chart(probabilities, config['data']['class_names'], prediction), use_column_width=True)

                # --- RECTIFICATION LOGIC (Moderate vs Very Mild) ---
                moderate_idx = config['data']['class_names'].index("ModerateDemented")
                very_mild_idx = config['data']['class_names'].index("VeryMildDemented")
                
                if class_name == "VeryMildDemented" and probabilities[moderate_idx] > 0.15:
                    st.warning("⚠️ **Potential Severity Underestimation**: The model detected 'Very Mild' but there is a non-trivial signature ({:.1f}%) for 'Moderate' dementia. This could be a misclassification of a more advanced stage.".format(probabilities[moderate_idx] * 100))
                    st.info("💡 **Clinical Insight**: Based on the Hybrid Model's attention map, look for significant atrophy in the hippocampus and cortical regions which are typical in Moderate stages.")
                elif class_name == "ModerateDemented" and confidence < 0.6:
                    st.info("💡 **Low Confidence Observation**: The model suggests Moderate dementia but with lower confidence. Comparison with CNN/ViT results is highly recommended.")


        # --- TAB 2: MODEL COMPARISON ---
        with tab2:
            st.markdown("### Architecture Comparison")
            st.info("Comparing inference results across three different architectures.")
    
            comp_cols = st.columns(3)
            
            models_to_run = [
                ('CNN (EfficientNet)', 'cnn'),
                ('Vision Transformer', 'vit'),
                ('Hybrid (CNN+ViT)', 'hybrid')
            ]
            
            comparison_results = []
            
            for idx, (name, arch) in enumerate(models_to_run):
                with comp_cols[idx]:
                    st.markdown(f"#### {name}")
                    
                    try:
                        m, d, c, p_path = get_model(arch)
                        if not p_path:
                             st.caption("⚠️ Weights not found")
                        
                        pred, probs = predict_image(image, m, d, c)
                        cls = get_class_name(pred, c)
                        conf = probs[pred]
                        
                        comparison_results.append({
                            "Architecture": name,
                            "Prediction": cls,
                            "Confidence": f"{conf*100:.1f}%",
                            "Status": "✅ Loaded" if p_path else "⚠️ Default"
                        })
                        
                        st.markdown(f"**{cls}**")
                        st.progress(float(conf))
                        st.caption(f"Confidence: {conf*100:.1f}%")
                        
                        # Mini chart
                        st.plotly_chart(plot_probability_chart(probs, c['data']['class_names'], pred), use_column_width=True)
                        
                    except Exception as e:
                        st.error(f"Failed to load {name}")
                        st.caption(str(e))
                        comparison_results.append({
                            "Architecture": name,
                            "Prediction": "Error",
                            "Confidence": "N/A",
                            "Status": "❌ Failed"
                        })

            st.markdown("---")
            st.markdown("#### Summary Table")
            st.table(comparison_results)

        # --- TAB 3: CLINICAL DETAILS ---
        with tab3:
            # Reusing the dictionary logic from original app
            diagnosis_info = {
                "NonDemented": {
                    "title": "Healthy Control",
                    "details": "No significant atrophy or ventricular enlargement detected.",
                    "rec": "Routine monitoring.",
                    "color": "#4ade80"
                },
                "VeryMildDemented": {
                    "title": "Very Mild Cognitive Impairment",
                    "details": "Subtle hippocampal shrinkage detected.",
                    "rec": "Consult neurologist; cognitive exercises recommended.",
                    "color": "#facc15"
                },
                "MildDemented": {
                    "title": "Mild Dementia",
                    "details": "Visible atrophy in temporal lobes.",
                    "rec": "Medical intervention and care planning advised.",
                    "color": "#fb923c"
                },
                "ModerateDemented": {
                    "title": "Moderate Dementia",
                    "details": "Widespread cortical atrophy.",
                    "rec": "Full-time care and safety measures required.",
                    "color": "#f87171"
                }
            }
            
            info = diagnosis_info.get(class_name, diagnosis_info["NonDemented"])
            
            st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid {info['color']}">
                <h3 style="color:{info['color']}">{info['title']}</h3>
                <p><strong>Clinical Observation:</strong> {info['details']}</p>
                <p><strong>Recommendation:</strong> {info['rec']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Probability Breakdown")
            for i, p in enumerate(probabilities):
                 label = config['data']['class_names'][i]
                 st.markdown(f"- **{label}**: {p*100:.2f}%")

            st.markdown("---")
            st.markdown("### 🤖 Diagnosis Agent")
            
            if st.button("Generate Comprehensive AI Report", help="Consult the AI Neurologist Agent for a detailed report"):
                with st.spinner("Consulting AI Specialist..."):
                    # Prepare Context
                    probs_dict = {config['data']['class_names'][i]: float(p) for i, p in enumerate(probabilities)}
                    
                    ctx = PredictionContext(
                        predicted_class=class_name,
                        confidence_score=float(confidence),
                        probabilities=probs_dict,
                        model_architecture="Hybrid EfficientNet+ViT"
                    )
                    
                    try:
                        report = asyncio.run(generate_clinical_report(ctx))
                        
                        st.markdown(f"#### 📝 Patient Summary")
                        st.write(report.patient_summary)
                        
                        st.markdown(f"#### 🔬 Detailed Findings")
                        st.write(report.detailed_findings)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### ⚠️ Risk Assessment")
                            st.info(report.risk_assessment)
                        with col2:
                            st.markdown("#### ✅ Recommended Actions")
                            for rec in report.recommended_actions:
                                st.markdown(f"- {rec}")
                                
                        with st.expander("❓ Questions for Specialist"):
                            for q in report.questions_for_specialist:
                                st.markdown(f"- {q}")
                                
                    except Exception as e:
                        st.error(f"Agent Error: {str(e)}")
                        st.warning("Please ensure OPENROUTER_API_KEY is set in environment variables or app/agent.py")

    else:
        st.markdown("""
        <div style="text-align: center; padding: 4rem; border: 2px dashed rgba(255,255,255,0.1); border-radius: 20px;">
            <h2>👋 Welcome to NeuralScan</h2>
            <p style="color: #94a3b8;">Upload a brain MRI scan to begin the multi-model analysis.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
