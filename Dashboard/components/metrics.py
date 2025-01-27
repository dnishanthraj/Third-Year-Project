import pandas as pd
import streamlit as st
import torch
import sys
from pathlib import Path

# Add Segmentation directory to sys.path
SEGMENTATION_DIR = Path(__file__).resolve().parents[2] / "Segmentation"
sys.path.append(str(SEGMENTATION_DIR))

# Import dice_coef and iou_score from Segmentation.metrics
from metrics import dice_coef2, iou_score

# Define color grading and contextual descriptions
def get_color_and_description_slice(metric_name, score):
    if metric_name == "Dice Score":
        if score >= 0.7:
            color = "background-color: #2ECC71; color: white;"  # Green
            description = "Excellent: The model performed very well, accurately predicting the majority of the region."
        elif 0.5 <= score < 0.7:
            color = "background-color: #F1C40F; color: black;"  # Yellow
            description = "Good: The model did decently, but missed some finer details of the target region."
        elif 0.3 <= score < 0.5:
            color = "background-color: #E67E22; color: white;"  # Orange
            description = "Fair: The model struggled with some areas, missing important parts of the region."
        else:
            color = "background-color: #E74C3C; color: white;"  # Red
            description = "Poor: The model failed to predict key areas of the region."
    elif metric_name == "IoU Score":
        if score >= 0.6:
            color = "background-color: #2ECC71; color: white;"  # Green
            description = "Excellent: High overlap between predicted and ground truth regions."
        elif 0.4 <= score < 0.6:
            color = "background-color: #F1C40F; color: black;"  # Yellow
            description = "Good: Reasonable overlap but with noticeable gaps."
        elif 0.2 <= score < 0.4:
            color = "background-color: #E67E22; color: white;"  # Orange
            description = "Fair: Low overlap, with substantial parts missing or incorrect."
        else:
            color = "background-color: #E74C3C; color: white;"  # Red
            description = "Poor: Minimal overlap; the model failed to predict accurately."
    return color, description

def display_scores_table(predicted_mask, ground_truth_mask):
    """Display a table with dynamically calculated Dice Score and IoU Score, including color and descriptions."""
    predicted_tensor = torch.tensor(predicted_mask)
    ground_truth_tensor = torch.tensor(ground_truth_mask)

    # Calculate Dice and IoU scores
    dice = dice_coef2(predicted_tensor, ground_truth_tensor).item()  # Ensure it's a Python float
    iou = iou_score(predicted_tensor, ground_truth_tensor).item()

    # Create and style the table with color gradients
    scores_data = {
        "Metric": ["Dice Score", "IoU Score"],
        "Value": [dice, iou],
    }
    scores_df = pd.DataFrame(scores_data)

    # Apply color grading and contextual descriptions
    styled_table = scores_df.style.format({"Value": "{:.3f}"}).apply(
        lambda x: [
            get_color_and_description_slice(x["Metric"], x["Value"])[0] if col == "Value" else ""
            for col in scores_df.columns
        ],
        axis=1,
    )

    # Display the table
    st.subheader("Segmentation Metrics with Explanations")
    st.table(styled_table)

    # Display contextual descriptions
    st.write("**Metric Explanations:**")
    for _, row in scores_df.iterrows():
        _, description = get_color_and_description_slice(row["Metric"], row["Value"])
        st.markdown(f"**{row['Metric']}:** {description}")

# Helper function: Color grading and descriptions for specific metrics
def get_color_and_description_overall(metric, score):
    if metric == "Dice":
        if score >= 0.7:
            color = "background-color: #2ECC71; color: white;"  # Green
            description = "Excellent segmentation performance with high overlap."
        elif 0.5 <= score < 0.7:
            color = "background-color: #F1C40F; color: black;"  # Yellow
            description = "Good performance but missing finer details."
        elif 0.3 <= score < 0.5:
            color = "background-color: #E67E22; color: white;"  # Orange
            description = "Fair performance; significant areas are missed."
        else:
            color = "background-color: #E74C3C; color: white;"  # Red
            description = "Poor performance with minimal overlap."
    elif metric == "IoU":
        if score >= 0.6:
            color = "background-color: #2ECC71; color: white;"
            description = "Excellent overlap between prediction and ground truth."
        elif 0.4 <= score < 0.6:
            color = "background-color: #F1C40F; color: black;"
            description = "Good overlap but room for improvement."
        elif 0.2 <= score < 0.4:
            color = "background-color: #E67E22; color: white;"
            description = "Fair overlap; model misses critical regions."
        else:
            color = "background-color: #E74C3C; color: white;"
            description = "Poor overlap with little correspondence."
    elif metric == "Precision":
        if score >= 0.75:
            color = "background-color: #2ECC71; color: white;"
            description = "High precision with minimal false positives."
        elif 0.5 <= score < 0.75:
            color = "background-color: #F1C40F; color: black;"
            description = "Moderate precision; false positives may be noticeable."
        else:
            color = "background-color: #E74C3C; color: white;"
            description = "Low precision with many false positives."
    elif metric == "Recall":
        if score >= 0.75:
            color = "background-color: #2ECC71; color: white;"
            description = "High recall with most true positives captured."
        elif 0.5 <= score < 0.75:
            color = "background-color: #F1C40F; color: black;"
            description = "Moderate recall; some true positives are missed."
        else:
            color = "background-color: #E74C3C; color: white;"
            description = "Low recall; significant true positives are missed."
    elif metric == "FPPS":
        if score <= 2.0:
            color = "background-color: #2ECC71; color: white;"
            description = "Low false positives per scan; strong model performance."
        elif 2.0 < score <= 4.0:
            color = "background-color: #F1C40F; color: black;"
            description = "Moderate false positives; manageable but not optimal."
        else:
            color = "background-color: #E74C3C; color: white;"
            description = "High false positives; potential over-segmentation."
    else:
        color = ""
        description = "No description available."
    return color, description