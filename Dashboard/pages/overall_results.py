import streamlit as st
import pandas as pd
import os
from components import calculate_precision, calculate_recall, calculate_fpps, get_color_and_description_overall, load_log_data

# Dynamically resolve the paths to the model_outputs folder
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_OUTPUTS_DIR = os.path.join(ROOT_DIR, "Project", "Segmentation", "model_outputs")

# List all folders under model_outputs
available_folders = [f for f in os.listdir(MODEL_OUTPUTS_DIR) if os.path.isdir(os.path.join(MODEL_OUTPUTS_DIR, f))]

# Add a multiselect to select folders for comparison
selected_folders = st.sidebar.multiselect("Select Folders to Compare", available_folders, default=[available_folders[0]] if available_folders else [])

# Predefined order of metrics
METRIC_ORDER = [
    "Dice", "IoU", "Total Slices", "Total Patients",
    "True Positive (TP)", "True Negative (TN)",
    "False Positive (FP)", "False Negative (FN)",
    "Precision", "Recall", "FPPS"
]

# Sidebar option to select metrics type
display_clean = st.sidebar.radio("Select Metrics Type", ["No Clean", "Clean"], key="metrics_type_radio")

# Helper function to process a single folder
def process_folder(folder, display_clean):
    metrics_file_path = os.path.join(MODEL_OUTPUTS_DIR, folder, "metrics", "metrics.csv")
    metrics_clean_file_path = os.path.join(MODEL_OUTPUTS_DIR, folder, "metrics", "metrics_clean.csv")
    log_file_path = os.path.join(MODEL_OUTPUTS_DIR, folder, "log.csv")

    try:
        metrics_data_no_clean = pd.read_csv(metrics_file_path)
        metrics_data_clean = pd.read_csv(metrics_clean_file_path)
        log_data = load_log_data(log_file_path)
    except FileNotFoundError:
        st.warning(f"Metrics or log file not found for {folder}.")
        return None, None

    if display_clean == "Clean":
        no_clean_dict = metrics_data_no_clean.set_index("Metric").to_dict()["Result"]
        clean_dict = metrics_data_clean.set_index("Metric").to_dict()["Result"]

        combined_metrics = {}
        combined_metrics["Total Slices"] = no_clean_dict["Total Slices"] + clean_dict["Total Slices"]
        combined_metrics["Total Patients"] = no_clean_dict["Total Patients"] + clean_dict["Total Patients"]
        combined_metrics["True Positive (TP)"] = no_clean_dict["True Positive (TP)"] + clean_dict["True Positive (TP)"]
        combined_metrics["True Negative (TN)"] = no_clean_dict["True Negative (TN)"] + clean_dict["True Negative (TN)"]
        combined_metrics["False Positive (FP)"] = no_clean_dict["False Positive (FP)"] + clean_dict["False Positive (FP)"]
        combined_metrics["False Negative (FN)"] = no_clean_dict["False Negative (FN)"] + clean_dict["False Negative (FN)"]

        # Recalculate metrics
        total_slices = combined_metrics["Total Slices"]
        combined_metrics["Dice"] = (
            no_clean_dict["Dice"] * no_clean_dict["Total Slices"]
            + clean_dict["Dice"] * clean_dict["Total Slices"]
        ) / total_slices
        combined_metrics["IoU"] = (
            no_clean_dict["IoU"] * no_clean_dict["Total Slices"]
            + clean_dict["IoU"] * clean_dict["Total Slices"]
        ) / total_slices
        combined_metrics["Precision"] = calculate_precision(
            combined_metrics["True Positive (TP)"], combined_metrics["False Positive (FP)"]
        )
        combined_metrics["Recall"] = calculate_recall(
            combined_metrics["True Positive (TP)"], combined_metrics["False Negative (FN)"]
        )
        combined_metrics["FPPS"] = calculate_fpps(
            combined_metrics["False Positive (FP)"], combined_metrics["Total Patients"]
        )

        metrics_data = pd.DataFrame(
            [{"Metric": k, "Result": v} for k, v in combined_metrics.items()]
        )
    else:
        metrics_data = metrics_data_no_clean

    metrics_data["Order"] = metrics_data["Metric"].apply(lambda x: METRIC_ORDER.index(x) if x in METRIC_ORDER else len(METRIC_ORDER))
    metrics_data = metrics_data.sort_values("Order").drop(columns=["Order"])
    metrics_data = metrics_data.set_index("Metric")
    metrics_data.columns = [folder]  # Rename column for the folder
    return metrics_data, log_data

# Process selected folders

if not selected_folders:
    st.title("Explainable XAI for Lung Nodule Segmentation - Overall Results")
    st.write("Please select a folder to view the results.")
comparison_data = []
logs_data = {}
for folder in selected_folders:
    folder_data, log_data = process_folder(folder, display_clean)
    if folder_data is not None:
        comparison_data.append(folder_data)
    if log_data is not None:
        logs_data[folder] = log_data

# Display Graph First
if logs_data:
    st.title("Explainable XAI for Lung Nodule Segmentation - Overall Results")
    st.subheader("Graph for Selected Metric Across Folders")
    graph_metric = st.selectbox("Select Metric to Graph", ["Loss", "IoU", "Dice"])

    # Toggle display for all training and validation curves
    show_training = st.checkbox("Show All Training Curves", value=True)
    show_validation = st.checkbox("Show All Validation Curves", value=True)

    combined_data = pd.DataFrame()
    for folder, log in logs_data.items():
        if graph_metric.lower() in log.columns and show_training:
            combined_data[folder + " - Training"] = log[f"{graph_metric.lower()}"]
        if f"val_{graph_metric.lower()}" in log.columns and show_validation:
            combined_data[folder + " - Validation"] = log[f"val_{graph_metric.lower()}"]
    if not combined_data.empty:
        st.line_chart(combined_data)

# Generate Combined Metrics Table with Explanations
if comparison_data:
    for folder, metrics_df in zip(selected_folders, comparison_data):
        # Add explanation for each metric
        metrics_with_explanations = []
        for metric, score in metrics_df[folder].items():
            color, description = get_color_and_description_overall(metric, score)
            metrics_with_explanations.append({"Metric": metric, "Result": score, "Description": description})

        metrics_with_explanations_df = pd.DataFrame(metrics_with_explanations)

        # Style the metrics table with color coding
        def apply_color(row):
            color, _ = get_color_and_description_overall(row["Metric"], row["Result"])
            return [color if col == "Result" else "" for col in row.index]

        styled_metrics = metrics_with_explanations_df.style.format({"Result": "{:.4f}"}).apply(apply_color, axis=1)

        # Display folder name and metrics table
        st.subheader(f"Folder: `{folder}`")
        st.write("Metrics with Explanations:")
        st.table(styled_metrics)
