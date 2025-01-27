import streamlit as st
import pandas as pd
import os
from components import *  # Ensure this imports your `load_log_data`

# Dynamically resolve the paths to the metrics files
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
METRICS_FILE_PATH = os.path.join(
    ROOT_DIR,
    "Project",
    "Segmentation",
    "model_outputs",
    "NestedUNET_with_augmentation_20250124_153410_92030151",
    "metrics",
    "metrics.csv",
)
METRICS_CLEAN_FILE_PATH = os.path.join(
    ROOT_DIR,
    "Project",
    "Segmentation",
    "model_outputs",
    "NestedUNET_with_augmentation_20250124_153410_92030151",
    "metrics",
    "metrics_clean.csv",
)

LOG_FILE_PATH = os.path.join(
    ROOT_DIR,
    "Project",
    "Segmentation",
    "model_outputs",
    "NestedUNET_with_augmentation_20250124_153410_92030151",
    "log.csv",
)

# Load log data (if log.csv is used for interactive plots)
log_data = load_log_data(LOG_FILE_PATH)

# Handle cases where log.csv is not found
if log_data is None:
    st.error(f"Log file not found: {LOG_FILE_PATH}")
else:
    # Display Title and Description for Log Data
    st.title("Explainable XAI for Lung Nodule Segmentation - Overall Results")
    st.write("Below is an interactive plot for loss, IoU, and Dice metrics over epochs.")

    # Sidebar options for metric and curve selection
    st.sidebar.subheader("Graph Options")
    metric_choice = st.sidebar.selectbox("Select Metric", ["Loss", "IoU", "Dice"])
    show_training = st.sidebar.checkbox("Show Training", value=True)
    show_validation = st.sidebar.checkbox("Show Validation", value=True)

    # Prepare data based on selection
    if metric_choice == "Loss":
        data = log_data[["epoch", "loss", "val_loss"]].rename(
            columns={"loss": "Training", "val_loss": "Validation"}
        )
    elif metric_choice == "IoU":
        data = log_data[["epoch", "iou", "val_iou"]].rename(
            columns={"iou": "Training", "val_iou": "Validation"}
        )
    elif metric_choice == "Dice":
        data = log_data[["epoch", "dice", "val_dice"]].rename(
            columns={"dice": "Training", "val_dice": "Validation"}
        )

    data = data.set_index("epoch")

    # Filter data based on toggles
    if show_training and show_validation:
        st.line_chart(data, use_container_width=True)
    elif show_training:
        st.line_chart(data[["Training"]], use_container_width=True)
    elif show_validation:
        st.line_chart(data[["Validation"]], use_container_width=True)
    else:
        st.warning("Please select at least one curve to display.")

# Load the appropriate metrics file based on sidebar selection
display_clean = st.sidebar.radio("Select Metrics Type", ["No Clean", "Clean"])

metrics_file = METRICS_CLEAN_FILE_PATH if display_clean == "Clean" else METRICS_FILE_PATH

try:
    metrics_data = pd.read_csv(metrics_file)
except FileNotFoundError:
    st.error(f"Metrics file not found: {metrics_file}")
    metrics_data = None

if metrics_data is not None:
    # Create a style function for color grading
    def apply_color(row):
        if row["Metric"] in ["Dice", "IoU", "Precision", "Recall", "FPPS"]:
            color, _ = get_color_and_description_overall(row["Metric"], row["Result"])
            return [color if col == "Result" else "" for col in row.index]
        return [""] * len(row)

    # Apply color grading
    styled_table = metrics_data.style.format({"Result": "{:.4f}"}).apply(
        apply_color,
        axis=1,
    )

    # Display Metrics Table
    st.subheader("Metrics Summary with Explanations")
    st.write(
        f"Below is the table summarizing the performance metrics for lung nodule segmentation ({display_clean})."
    )
    st.table(styled_table)

    # Display explanations only for relevant metrics
    st.write("**Metric Explanations:**")
    for _, row in metrics_data.iterrows():
        if row["Metric"] in ["Dice", "IoU", "Precision", "Recall", "FPPS"]:
            _, description = get_color_and_description_overall(row["Metric"], row["Result"])
            st.markdown(f"- **{row['Metric']}:** {description}")
else:
    st.warning("Unable to display metrics summary as the file could not be loaded.")
