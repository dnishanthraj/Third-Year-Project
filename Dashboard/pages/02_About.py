import streamlit as st
import os

# Configure the page
st.set_page_config(page_title="About", layout="wide")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
    /* Force Poppins on every element */
    [data-testid="stAppViewContainer"] *, [data-testid="stSidebar"] * {
        font-family: 'Poppins', sans-serif !important;
    }
    .block-container {
        max-width: 1400px !important;  /* or 1600px, adjust to taste */
        padding: 3rem 2rem !important; /* Adjust as needed for your taste */
    }
                /* Increase vertical spacing between nav items */
    /* (Optional) If you want to reduce vertical margin inside sidebar expanders: */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        margin: 0.2rem 0 !important;
    }        
    # [data-testid="stSidebarNav"] ul {
    #     margin-top: 0.5rem; /* space above the list */
    # }
    [data-testid="stSidebarNav"] ul li {
        margin-bottom: 0.2rem; /* space between items */
    }
    /* Add some padding around each link */
    [data-testid="stSidebarNav"] ul li a {
        padding: 0.3rem 1rem; 
        border-radius: 4px;
    }
            /* Increase spacing before h2 headings in the sidebar */
    [data-testid="stSidebar"] h2 {
        margin-bottom: 1rem !important;
    }      
    </style>
    """, unsafe_allow_html=True)

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
