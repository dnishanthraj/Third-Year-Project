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

# Sidebar toggle for clean or no-clean metrics
display_clean = st.sidebar.radio("Select Metrics Type", ["No Clean", "Clean"])

# Load the appropriate metrics file based on the toggle
if display_clean == "No Clean":
    metrics_file = METRICS_FILE_PATH
else:
    metrics_file = METRICS_CLEAN_FILE_PATH

try:
    metrics_data = pd.read_csv(metrics_file)
except FileNotFoundError:
    st.error(f"Metrics file not found: {metrics_file}")
    metrics_data = None

if metrics_data is not None:
    # Extract data from the selected metrics file
    metrics_summary = {
        "Metric": metrics_data["Metric"].tolist(),
        "Result": metrics_data["Result"].tolist(),
    }
    metrics_df = pd.DataFrame(metrics_summary)

    # Display Metrics Table
    st.subheader("Metrics Summary")
    st.write(
        f"Below is the table summarizing the overall performance metrics for lung nodule segmentation ({display_clean})."
    )
    st.table(metrics_df)

    # Add explanations or comments
    st.write("**Key Highlights:**")
    st.write("- **Dice Score** and **IoU** provide insights into segmentation performance.")
    st.write("- **FPPS (False Positives per Scan)** evaluates false positives normalized by patient count.")
else:
    st.warning("Unable to display metrics summary as the file could not be loaded.")
