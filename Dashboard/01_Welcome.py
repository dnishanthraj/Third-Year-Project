# ---------------------------------------------
# Welcome Page - Explainable AI Lung Nodule Segmentation
# ---------------------------------------------

import streamlit as st
import os

# ---------------------------------------------
# Page Configuration
# ---------------------------------------------
st.set_page_config(page_title="Welcome & About", layout="wide")

# ---------------------------------------------
# Custom CSS Styling
# ---------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
<style>
/* Force Poppins font on all elements */
[data-testid="stAppViewContainer"] *, [data-testid="stSidebar"] * {
    font-family: 'Poppins', sans-serif !important;
}
/* Adjust block container width and padding */
.block-container {
    max-width: 1400px !important;
    padding: 3rem 2rem !important;
}
/* Increase vertical spacing between sidebar items */
[data-testid="stSidebar"] .streamlit-expanderHeader {
    margin: 0.2rem 0 !important;
}
[data-testid="stSidebarNav"] ul li {
    margin-bottom: 0.2rem;
}
/* Add padding and border radius to sidebar links */
[data-testid="stSidebarNav"] ul li a {
    padding: 0.3rem 1rem;
    border-radius: 4px;
}
/* Adjust sidebar section heading margins */
[data-testid="stSidebar"] h2 {
    margin-bottom: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# Main Title
# ---------------------------------------------
st.title("Explainable AI Lung Nodule Segmentation Dashboard")
st.markdown("---")

# ---------------------------------------------
# Welcome Section
# ---------------------------------------------
st.markdown("""
Welcome to the Explainable AI Lung Nodule Segmentation Dashboard!  
Please select an option from the sidebar to continue.
""")
st.markdown("---")

# ---------------------------------------------
# About Section - Project Overview
# ---------------------------------------------
st.header("Project Overview")
st.markdown("""
Lung cancer remains one of the leading causes of cancer-related deaths worldwide, making early detection essential.  
This dashboard was developed to assist radiologists by automating the segmentation of lung nodules from CT scans using advanced deep learning models.  
It addresses the challenges of accurately identifying and segmenting small, often elusive nodules, providing a platform for model comparison and performance evaluation.
""")

# ---------------------------------------------
# About Section - Key Contributions
# ---------------------------------------------
st.header("Key Contributions")
st.markdown("""
- **Automated Segmentation Models:**  
  Implementation of deep learning architectures (e.g., U-Net, U-Net++) to detect and segment lung nodules.

- **False Positive Reduction (FPR):**  
  Integration of an XGBoost-based classifier to filter out spurious detections, ensuring more reliable results.

- **Explainability with Grad-CAM:**  
  Utilization of Grad-CAM heatmaps to visualize the regions influencing model predictions, enhancing transparency and trust.

- **Comprehensive Performance Evaluation:**  
  Evaluation using metrics such as Dice Score, IoU, Precision, Recall, Specificity, Accuracy, and F1-Score to quantify model effectiveness.

- **Interactive Dashboard:**  
  A user-friendly interface that allows radiologists to compare models, inspect individual CT slices, and review detailed performance metrics.
""")

# ---------------------------------------------
# About Section - Intended Use and Purpose
# ---------------------------------------------
st.header("Intended Use and Purpose")
st.markdown("""
The Lung Nodule Segmentation Dashboard is designed to:
- **Enhance Diagnostic Accuracy:**  
  Provide automated segmentation and reliable performance metrics to aid radiologists in detecting lung nodules accurately.

- **Improve Workflow Efficiency:**  
  Reduce manual workload by automating the segmentation process and offering clear, visual insights through explainability tools.

- **Facilitate Model Comparison:**  
  Allow researchers and clinicians to compare multiple segmentation models side-by-side and select the most effective approach.

- **Promote Transparency in AI:**  
  Use Grad-CAM visualizations and detailed evaluations to make the model's decision-making process clear and interpretable.
""")

# ---------------------------------------------
# Final Notes
# ---------------------------------------------
st.markdown("---")
st.markdown("For further technical details and operational guidance, please refer to the 'Guide' section of the dashboard.")
