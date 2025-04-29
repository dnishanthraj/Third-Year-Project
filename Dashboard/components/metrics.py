# -------------------------------------------------------
# Metrics Calculation, Color Grading, and Statistical Testing Utilities
# -------------------------------------------------------

import pandas as pd
import streamlit as st
import torch
import sys
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from pathlib import Path

# -------------------------------------------------------
# Add Segmentation directory to sys.path for custom metric imports
# -------------------------------------------------------
SEGMENTATION_DIR = Path(__file__).resolve().parents[2] / "Segmentation"
sys.path.append(str(SEGMENTATION_DIR))

# -------------------------------------------------------
# Import metrics functions from Segmentation module
# -------------------------------------------------------
from metrics import dice_coef2, iou_score, calculate_precision, calculate_recall, calculate_fpps

# -------------------------------------------------------
# Slice-level Color Grading and Descriptions
# -------------------------------------------------------
def get_color_and_description_slice(metric_name, score):
    """Return background color and description based on slice-level Dice or IoU score."""
    if metric_name == "Dice":
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
    elif metric_name == "IoU":
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

# -------------------------------------------------------
# Display Dice and IoU Scores for a Single Slice
# -------------------------------------------------------
def display_scores_table(predicted_mask, ground_truth_mask):
    """Display a styled table with Dice and IoU scores and contextual feedback."""
    predicted_tensor = torch.tensor(predicted_mask)
    ground_truth_tensor = torch.tensor(ground_truth_mask)

    # Calculate metrics
    dice = dice_coef2(predicted_tensor, ground_truth_tensor).item()
    iou = iou_score(predicted_tensor, ground_truth_tensor).item()

    # Prepare table data
    metrics_data = []
    for metric_name, score in [("Dice", dice), ("IoU", iou)]:
        _, description = get_color_and_description_slice(metric_name, score)
        metrics_data.append({"Metric": metric_name, "Result": score, "Description": description})

    metrics_df = pd.DataFrame(metrics_data)

    # Apply color formatting
    def apply_color(row):
        color, _ = get_color_and_description_slice(row["Metric"], row["Result"])
        return [color if col == "Result" else "" for col in row.index]

    styled_table = metrics_df.style.format({"Result": "{:.4f}"}).apply(
        lambda row: apply_color(row), axis=1
    )

    st.table(styled_table)

# -------------------------------------------------------
# Overall-level Color Grading and Descriptions
# -------------------------------------------------------
def get_color_and_description_overall(metric, score):
    """Return background color and description based on overall model metrics."""
    if metric == "Dice":
        if score >= 0.7:
            color = "background-color: #2ECC71; color: white;"
            description = "Excellent segmentation performance with high overlap."
        elif 0.5 <= score < 0.7:
            color = "background-color: #F1C40F; color: black;"
            description = "Good performance but missing finer details."
        elif 0.3 <= score < 0.5:
            color = "background-color: #E67E22; color: white;"
            description = "Fair performance; significant areas are missed."
        else:
            color = "background-color: #E74C3C; color: white;"
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
            description = "Moderate recall; some true positives missed."
        else:
            color = "background-color: #E74C3C; color: white;"
            description = "Low recall; many true positives missed."

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

    elif metric == "Accuracy":
        if score >= 0.8:
            color = "background-color: #2ECC71; color: white;"
            description = "High overall correctness."
        elif 0.6 <= score < 0.8:
            color = "background-color: #F1C40F; color: black;"
            description = "Moderate correctness; room for improvement."
        elif 0.4 <= score < 0.6:
            color = "background-color: #E67E22; color: white;"
            description = "Fair correctness; model struggles in many cases."
        else:
            color = "background-color: #E74C3C; color: white;"
            description = "Low correctness; model frequently misclassifies."

    elif metric == "Specificity":
        if score >= 0.8:
            color = "background-color: #2ECC71; color: white;"
            description = "High specificity; few false alarms."
        elif 0.6 <= score < 0.8:
            color = "background-color: #F1C40F; color: black;"
            description = "Moderate specificity; false positives noticeable."
        elif 0.4 <= score < 0.6:
            color = "background-color: #E67E22; color: white;"
            description = "Fair specificity; many false positives present."
        else:
            color = "background-color: #E74C3C; color: white;"
            description = "Low specificity; frequent false positives."

    elif metric == "F1-Score":
        if score >= 0.75:
            color = "background-color: #2ECC71; color: white;"
            description = "High F1-score; good balance of precision and recall."
        elif 0.5 <= score < 0.75:
            color = "background-color: #F1C40F; color: black;"
            description = "Moderate F1-score; decent balance but can improve."
        elif 0.3 <= score < 0.5:
            color = "background-color: #E67E22; color: white;"
            description = "Fair F1-score; many missed predictions."
        else:
            color = "background-color: #E74C3C; color: white;"
            description = "Low F1-score; poor precision and recall."

    else:
        color = ""
        description = "No description available."

    return color, description