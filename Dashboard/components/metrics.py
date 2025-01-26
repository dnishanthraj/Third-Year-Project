import pandas as pd
import streamlit as st
import torch
import sys
from pathlib import Path

# Add Segmentation directory to sys.path
SEGMENTATION_DIR = Path(__file__).resolve().parents[2] / "Segmentation"
sys.path.append(str(SEGMENTATION_DIR))

# Import dice_coef and iou_score from Segmentation.metrics
from metrics import dice_coef, iou_score

def display_scores_table(predicted_mask, ground_truth_mask):
    """Display a table with dynamically calculated Dice Score and IoU Score."""
    predicted_tensor = torch.tensor(predicted_mask)
    ground_truth_tensor = torch.tensor(ground_truth_mask)
    
    # Calculate Dice and IoU scores
    dice = dice_coef(predicted_tensor, ground_truth_tensor)  # Rename the variable
    iou = iou_score(predicted_tensor, ground_truth_tensor)   # Rename the variable

    # Create and display the table
    scores_data = {
        "Metric": ["Dice Score", "IoU Score"],
        "Value": [dice, iou],
    }
    scores_df = pd.DataFrame(scores_data)
    st.subheader("Segmentation Metrics")
    st.table(scores_df)
