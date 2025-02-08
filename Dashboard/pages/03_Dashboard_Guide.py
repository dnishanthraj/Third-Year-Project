import streamlit as st
import os

# Configure the page
st.set_page_config(page_title="Dashboard Guide", layout="wide")

st.title("Dashboard Guide")
st.markdown("---")
st.markdown("""
This guide explains the key concepts, metrics, and functionality of the Lung Nodule Segmentation Dashboard. Whether you are new to machine learning or simply need a refresher on clinical metrics, this page is here to help.
""")

# Using the Dashboard Section
st.header("Using the Dashboard")
st.markdown("""
The dashboard features a sidebar with quick access to several pages:

- **Slice Viewer:**  
  View each model's individual predictions, including Grad-CAM heatmaps. You can also annotate images, save your annotations, and review individual Dice and IoU scores.

- **Overall Results:**  
  Compare training and validation curves for each model, examine all performance metrics, and assess the effect of the False Positive Reduction (FPR) system by comparing scans with and without nodules.

- **Support:**  
  Access a help form to submit queries and view frequently asked questions (FAQ).
""")

st.header("Key Terminology")

with st.expander("Model & Segmentation Overview"):
    st.markdown("""
**Model:**  
A model is a computer program that learns from examples. In this dashboard, the model is trained on medical scans to recognize lung nodules.

**Segmentation:**  
Segmentation is the process of dividing an image into meaningful regions. Here, it refers to generating a **predicted mask** that highlights areas where nodules are detected.  
- **Predicted Mask:** The area highlighted by the model as containing a nodule.  
- **Ground Truth:** The actual nodule regions, as identified by medical experts.

This helps radiologists by automatically marking potential areas of concern, streamlining diagnosis and review.
    """)
    overview_img = os.path.join(os.path.dirname(__file__), "..", "figures", "model_segmentation.png")
    if os.path.exists(overview_img):
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(overview_img, caption="Model and Segmentation Overview", use_column_width=False, width=600)
        st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Training and Validation Curves & Epochs"):
    st.markdown("""
**Training and Validation Curves** show how the model learns over time.  
An **epoch** is one complete pass through the entire training dataset.  
- **Training Curve:** Displays how well the model performs on the data it is learning from.
- **Validation Curve:** Shows how well the model performs on new, unseen data.

A well-trained model will show both curves improving and then leveling off. If the training curve keeps improving while the validation curve stalls or worsens, it suggests **overfitting**, meaning that the model is memorizing the training data rather than learning generalizable patterns. Overfitting is dangerous in clinical applications since it means the model may not perform reliably on new scans.
    """)
    curve_img = os.path.join(os.path.dirname(__file__), "..", "figures", "curve.png")
    if os.path.exists(curve_img):
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(curve_img, caption="Training vs. Validation Curves", use_column_width=False, width=600)
        st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Dice Score"):
    st.markdown("""
The **Dice Score** measures the overlap between the predicted nodule area and the actual nodule area.  
- **Range:** 0 (no overlap) to 1 (perfect overlap).  
- **1.0:** Perfect overlap.  
- **0.0:** No overlap.  

For lung nodule segmentation, a Dice Score above 0.7 is generally considered good. A higher Dice Score means the model is accurately outlining the nodules, which is crucial for reliable diagnosis.
    """)
    dice_img = os.path.join(os.path.dirname(__file__), "..", "figures", "dice.png")
    if os.path.exists(dice_img):
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(dice_img, caption="Dice Score Illustration", use_column_width=False, width=600)
        st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Intersection over Union (IoU)"):
    st.markdown("""
**Intersection over Union (IoU)** also measures the overlap between the predicted segmentation and the actual nodule region.  
- **Range:** 0 to 1.  
- **1.0:** A perfect match.  
- **0.0:** No overlap.
- **Above 0.7:** Indicates good segmentation.
- **Between 0.3 and 0.5:** Indicates poor segmentation.

A high IoU means the model is accurately capturing the nodule without too many false positives or negatives.
    """)
    iou_img = os.path.join(os.path.dirname(__file__), "..", "figures", "iou.png")
    if os.path.exists(iou_img):
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(iou_img, caption="IoU Illustration", use_column_width=False, width=600)
        st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Confusion Matrix & Performance Metrics"):
    st.markdown("""
The **Confusion Matrix** helps break down the model's predictions into four parts:
- **True Positive (TP):** The model correctly identifies a nodule.
- **False Positive (FP):** The model incorrectly labels a non-nodule as a nodule.
- **False Negative (FN):** The model misses a nodule that is present.
- **True Negative (TN):** The model correctly identifies areas without nodules.

From these, we calculate several performance metrics:
1. **Precision (Positive Predictive Value):**  
   - **Calculation:** TP / (TP + FP)  
   - **Meaning:** High precision (ideally above 0.7) means fewer false alarms.
2. **Recall (Sensitivity):**  
   - **Calculation:** TP / (TP + FN)  
   - **Meaning:** High recall (typically above 0.7) means most nodules are detected.
3. **Specificity (True Negative Rate):**  
   - **Calculation:** TN / (TN + FP)  
   - **Meaning:** High specificity (above 0.7) indicates non-nodule areas are correctly identified.
4. **Accuracy:**  
   - **Calculation:** (TP + TN) / (TP + TN + FP + FN)  
   - **Note:** Can be misleading if nodules are rare compared to non-nodule areas.
5. **F1-Score:**  
   - **Calculation:** 2 * (Precision * Recall) / (Precision + Recall)  
   - **Meaning:** A balanced measure that considers both precision and recall.
    """)
    cm_img = os.path.join(os.path.dirname(__file__), "..", "figures", "confusion_matrix.png")
    if os.path.exists(cm_img):
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(cm_img, caption="Confusion Matrix Example", use_column_width=False, width=600)
        st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Grad-CAM (Model Explainability)"):
    st.markdown("""
**Grad-CAM (Gradient-weighted Class Activation Mapping)** is a technique used to visualize which parts of an image influenced the model's decision.  
- **How It Works:** It creates a heatmap overlay on the original scan.  
- **Color Interpretation:**  
  - **Red/Orange:** Areas that had the highest influence on the model's decision.
  - **Blue/Green:** Areas that had less influence.
  
This visualization helps radiologists understand and verify what the model is focusing on during segmentation.
    """)
    grad_cam_img = os.path.join(os.path.dirname(__file__), "..", "figures", "grad_cam.png")
    if os.path.exists(grad_cam_img):
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(grad_cam_img, caption="Grad-CAM Heatmap", use_column_width=False, width=600)
        st.markdown("</div>", unsafe_allow_html=True)

with st.expander("False Positive Reduction (FPR) and Clean Set"):
    st.markdown("""
**False Positive Reduction (FPR):**  
This system uses an additional classifier (for example, an XGBoost classifier) to filter out predictions that are likely false positives. In other words, it helps eliminate cases where the model incorrectly identifies a non-nodule as a nodule.

**Clean Set:**  
The Clean Set is a collection of scans that are known to have no nodules at all. This set is used to assess and improve the FPR system.  
- **Purpose:** Ensuring that the FPR system accurately identifies scans without nodules minimizes unnecessary alarms and follow-up tests.

By combining these, the dashboard ensures that only the most reliable detections are flagged, improving overall diagnostic confidence.
    """)
    fpr_img = os.path.join(os.path.dirname(__file__), "..", "figures", "fpr_clean_set.png")
    if os.path.exists(fpr_img):
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(fpr_img, caption="FPR and Clean Set Illustration", use_column_width=False, width=600)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")