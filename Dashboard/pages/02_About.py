import streamlit as st
import os

# Configure the page
st.set_page_config(page_title="About", layout="wide")

st.title("About the Lung Nodule Segmentation Dashboard")
st.markdown("---")

st.header("Project Overview")
st.markdown("""
Lung cancer remains one of the leading causes of cancer-related deaths worldwide, making early detection essential. 
This dashboard was developed to assist radiologists by automating the segmentation of lung nodules from CT scans using advanced deep learning models. 
It addresses the challenges of accurately identifying and segmenting small, often elusive nodules, providing a platform for model comparison and performance evaluation.
""")

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

st.markdown("---")
st.markdown("For further technical details and operational guidance, please refer to the 'Guide' section of the dashboard.")
