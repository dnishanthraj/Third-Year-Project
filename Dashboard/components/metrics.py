# components/metrics.py
import pandas as pd
import streamlit as st

def display_scores_table(dice_score=0.0, iou_score=0.0):
    """Display a table with Dice Score and IoU Score."""
    scores_data = {
        "Metric": ["Dice Score", "IoU Score"],
        "Value": [dice_score, iou_score],
    }
    scores_df = pd.DataFrame(scores_data)
    st.subheader("Segmentation Metrics")
    st.table(scores_df)
