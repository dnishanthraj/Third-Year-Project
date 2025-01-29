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

# Add Segmentation directory to sys.path
SEGMENTATION_DIR = Path(__file__).resolve().parents[2] / "Segmentation"
sys.path.append(str(SEGMENTATION_DIR))

# Import dice_coef and iou_score from Segmentation.metrics
from metrics import dice_coef2, iou_score, calculate_precision, calculate_recall, calculate_fpps

# Define color grading and contextual descriptions
def get_color_and_description_slice(metric_name, score):
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

def display_scores_table(predicted_mask, ground_truth_mask):
    """Display a table with dynamically calculated Dice Score and IoU Score, including descriptions."""
    predicted_tensor = torch.tensor(predicted_mask)
    ground_truth_tensor = torch.tensor(ground_truth_mask)

    # Calculate Dice and IoU scores
    dice = dice_coef2(predicted_tensor, ground_truth_tensor).item()  # Ensure it's a Python float
    iou = iou_score(predicted_tensor, ground_truth_tensor).item()

    # Prepare data with metrics, values, and descriptions
    metrics_data = []
    for metric_name, score in [("Dice", dice), ("IoU", iou)]:
        _, description = get_color_and_description_slice(metric_name, score)
        metrics_data.append({"Metric": metric_name, "Result": score, "Description": description})

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_data)

    # Styling function for the "Result" column
    def apply_color(row):
        color, _ = get_color_and_description_slice(row["Metric"], row["Result"])
        return [color if col == "Result" else "" for col in row.index]

    # Apply styling
    styled_table = metrics_df.style.format({"Result": "{:.4f}"}).apply(
        lambda row: apply_color(row), axis=1
    )

    # Display the table
    st.subheader("Segmentation Metrics with Explanations")
    st.table(styled_table)


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

def run_t_test(series_a, series_b):
    """
    Performs a two-sided independent t-test between two 1D arrays of data.
    Returns the test statistic and p-value.
    """
    t_stat, p_val = stats.ttest_ind(series_a, series_b, equal_var=False)
    return t_stat, p_val

def run_anova(data_dict):
    """
    Runs a one-way ANOVA across 3+ groups.
    data_dict: dict where keys are group labels, values are lists/arrays of data.
    
    Returns:
       - anova_results: (F-statistic, p-value)
       - posthoc_df: a DataFrame containing post-hoc (Tukey HSD) results 
                     if p < 0.05, otherwise None
    """
    # 1) Convert data_dict to a "long" DataFrame => columns=["value", "group"]
    all_values = []
    all_groups = []
    for group_name, values in data_dict.items():
        all_values.extend(values)
        all_groups.extend([group_name]*len(values))

    df_long = pd.DataFrame({"value": all_values, "group": all_groups})

    # 2) Fit OLS model and run ANOVA
    model = ols("value ~ C(group)", data=df_long).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # Get F and p-value from the ANOVA table
    F_stat = anova_table["F"][0]
    p_val = anova_table["PR(>F)"][0]
    
    # 3) If ANOVA is significant, run post-hoc (Tukey HSD)
    posthoc_df = None
    if p_val < 0.05:
        tukey = pairwise_tukeyhsd(endog=df_long["value"], groups=df_long["group"], alpha=0.05)
        posthoc_df = pd.DataFrame(data=tukey.summary()[1:], columns=tukey.summary()[0])
    
    return (F_stat, p_val), posthoc_df

def run_statistical_tests(metric_name, data_dict):
    """
    Main function to decide whether to run a t-test or an ANOVA+posthoc 
    for a given metric based on how many folders/groups are in data_dict.

    data_dict: { folder_name: [values_for_that_folder], ... }

    Returns a dictionary or DataFrame with the relevant stats results.
    """

    n_groups = len(data_dict)

    if n_groups < 2:
        return None  # Not enough groups to compare

    if n_groups == 2:
        # Only two folders => t-test
        group_names = list(data_dict.keys())
        series_a = data_dict[group_names[0]]
        series_b = data_dict[group_names[1]]
        t_stat, p_val = run_t_test(series_a, series_b)
        return {
            "metric": metric_name,
            "test": "t-test",
            "group1": group_names[0],
            "group2": group_names[1],
            "t_stat": t_stat,
            "p_value": p_val
        }
    else:
        # 3+ folders => ANOVA + possible Tukey post-hoc
        (F_stat, p_val), posthoc_df = run_anova(data_dict)
        result = {
            "metric": metric_name,
            "test": "ANOVA",
            "F_stat": F_stat,
            "p_value": p_val,
            "posthoc": posthoc_df  # can be None if not significant
        }
        return result