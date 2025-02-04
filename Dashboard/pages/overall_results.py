import streamlit as st
import pandas as pd
import os
import altair as alt
import collections

from components import (
    calculate_precision, calculate_recall, calculate_fpps,
    get_color_and_description_overall, load_log_data, lighten_color, run_statistical_tests
)

# Dynamically resolve the paths to the model_outputs folder
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_OUTPUTS_DIR = os.path.join(ROOT_DIR, "Project", "Segmentation", "model_outputs")

# List all folders under model_outputs
available_folders = [
    f for f in os.listdir(MODEL_OUTPUTS_DIR)
    if os.path.isdir(os.path.join(MODEL_OUTPUTS_DIR, f))
]

# Add a multiselect to select folders for comparison
selected_folders = st.sidebar.multiselect(
    "Select Folders to Compare",
    available_folders,
    default=[available_folders[0]] if available_folders else []
)

# Let the user pick colors for each folder
colors_for_folders = {}
for folder in selected_folders:
    default_color = "#008000"  # pick a nicer default than pure blue
    picked_color = st.sidebar.color_picker(
        f"Pick a color for: {folder}", 
        default_color, 
        key=f"color_{folder}"
    )
    colors_for_folders[folder] = picked_color

# Predefined order of metrics
METRIC_ORDER = [
    "Dice", "IoU", "Total Slices", "Total Patients",
    "True Positive (TP)", "True Negative (TN)",
    "False Positive (FP)", "False Negative (FN)",
    "Precision", "Recall", "FPPS"
]

# Sidebar option to select metrics type (i.e. combine clean data or not)
display_clean = st.sidebar.radio("Select Metrics Type", ["No Clean", "Clean"], key="metrics_type_radio")

# NEW: Sidebar option to select the metrics source: either Raw segmentation or FPR postprocessed.
metrics_source = st.sidebar.radio("Select Metrics Source", ["Raw", "FPR"], key="metrics_source_radio")

# -------------------------------------------------------------------
# Helper function: process_folder
# -------------------------------------------------------------------
def process_folder(folder, display_clean, metrics_source):
    """
    Reads the metrics files and log.csv from a folder.
    If metrics_source == "FPR", then load the files:
        metrics_fpr.csv and metrics_fpr_clean.csv.
    Otherwise, load metrics.csv and metrics_clean.csv.
    If display_clean == "Clean", combine the clean and non-clean metrics.
    Returns:
      - A DataFrame (indexed by Metric) with one column named as the folder.
      - The log DataFrame.
    """
    if metrics_source == "FPR":
        metrics_file_path = os.path.join(MODEL_OUTPUTS_DIR, folder, "metrics", "metrics_fpr.csv")
        metrics_clean_file_path = os.path.join(MODEL_OUTPUTS_DIR, folder, "metrics", "metrics_fpr_clean.csv")
    else:
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
            combined_metrics["True Positive (TP)"],
            combined_metrics["False Positive (FP)"],
        )
        combined_metrics["Recall"] = calculate_recall(
            combined_metrics["True Positive (TP)"],
            combined_metrics["False Negative (FN)"]
        )
        combined_metrics["FPPS"] = calculate_fpps(
            combined_metrics["False Positive (FP)"],
            combined_metrics["Total Patients"]
        )

        metrics_data = pd.DataFrame(
            [{"Metric": k, "Result": v} for k, v in combined_metrics.items()]
        )
    else:
        metrics_data = metrics_data_no_clean

    metrics_data["Order"] = metrics_data["Metric"].apply(
        lambda x: METRIC_ORDER.index(x) if x in METRIC_ORDER else len(METRIC_ORDER)
    )
    metrics_data = metrics_data.sort_values("Order").drop(columns=["Order"]).set_index("Metric")
    metrics_data.columns = [folder]
    return metrics_data, log_data

# -------------------------------------------------------------------
# Main Layout
# -------------------------------------------------------------------

st.title("Explainable XAI for Lung Nodule Segmentation - Overall Results")

if not selected_folders:
    st.write("Please select at least one folder from the sidebar to view results.")
    st.stop()

comparison_data = []
logs_data = {}

for folder in selected_folders:
    folder_data, log_data = process_folder(folder, display_clean, metrics_source)
    if folder_data is not None:
        comparison_data.append(folder_data)
    if log_data is not None:
        logs_data[folder] = log_data

# Plot training/validation curves if available
if logs_data:
    st.subheader("**Training & Validation Curves**")
    graph_metric = st.selectbox("Select Metric to Graph", ["Loss", "IoU", "Dice"])
    show_training = st.checkbox("Show All Training Curves", value=True)
    show_validation = st.checkbox("Show All Validation Curves", value=True)

    layers = []
    for folder, log in logs_data.items():
        if graph_metric.lower() not in log.columns and f"val_{graph_metric.lower()}" not in log.columns:
            continue

        folder_training_color = colors_for_folders.get(folder, "#000000")
        folder_validation_color = lighten_color(folder_training_color, factor=0.5)

        chart_data = pd.DataFrame({"epoch": log["epoch"]})
        if show_training and graph_metric.lower() in log.columns:
            chart_data["training"] = log[graph_metric.lower()]
        if show_validation and f"val_{graph_metric.lower()}" in log.columns:
            chart_data["validation"] = log[f"val_{graph_metric.lower()}"]

        melted_data = chart_data.melt(id_vars="epoch", var_name="type", value_name="value")

        training_chart = (
            alt.Chart(melted_data)
            .transform_filter("datum.type == 'training'")
            .mark_line(point=True)
            .encode(
                x=alt.X("epoch:Q", title="Epoch"),
                y=alt.Y("value:Q", title=graph_metric),
                color=alt.value(folder_training_color),
                tooltip=["epoch", "value"]
            )
        )

        validation_chart = (
            alt.Chart(melted_data)
            .transform_filter("datum.type == 'validation'")
            .mark_line(point=True)
            .encode(
                x=alt.X("epoch:Q", title="Epoch"),
                y=alt.Y("value:Q", title=graph_metric),
                color=alt.value(folder_validation_color),
                tooltip=["epoch", "value"]
            )
        )

        folder_chart = alt.layer(training_chart, validation_chart).properties(title=folder)
        layers.append(folder_chart)

    if layers:
        final_chart = alt.layer(*layers).interactive()
        st.altair_chart(final_chart, use_container_width=True)
    else:
        st.write("No valid training/validation columns found to plot.")

# Combine final metrics across folders
if comparison_data:
    final_metrics_df = pd.concat(comparison_data, axis=1)
    st.subheader("**Test Results Graph**")
    test_metrics_options = ["Dice", "IoU", "Precision", "Recall", "FPPS"]
    test_metrics_options = [m for m in test_metrics_options if m in final_metrics_df.index]

    if test_metrics_options:
        selected_test_metric = st.selectbox("Select a Test Metric to Graph", test_metrics_options)
        selected_metric_series = final_metrics_df.loc[selected_test_metric]
        df_scatter = pd.DataFrame({
            "Folder": selected_metric_series.index,
            "Value": selected_metric_series.values
        })
        domain_folders = list(colors_for_folders.keys())
        range_colors  = list(colors_for_folders.values())
        scatter_chart = (
            alt.Chart(df_scatter)
            .mark_circle(size=100)
            .encode(
                x=alt.X("Folder:N", axis=alt.Axis(title=None, labels=False, ticks=False)),
                y=alt.Y("Value:Q", title=selected_test_metric),
                color=alt.Color(
                    "Folder:N",
                    scale=alt.Scale(domain=domain_folders, range=range_colors),
                    legend=alt.Legend(title="Folder")
                ),
                tooltip=["Folder:N", "Value:Q"]
            )
            .properties(
                width=600,
                height=400,
                title=f"{selected_test_metric} Across Folders (Scatter)"
            )
        )
        st.altair_chart(scatter_chart, use_container_width=True)
    else:
        st.warning("No final metrics found to plot (Dice, IoU, etc. missing).")

    st.subheader("**Folder Metrics (Raw & FPR)**")
    for folder, metrics_df in zip(selected_folders, comparison_data):
        metrics_with_explanations = []
        for metric, score in metrics_df[folder].items():
            color, description = get_color_and_description_overall(metric, score)
            metrics_with_explanations.append({
                "Metric": metric,
                "Result": score,
                "Description": description
            })
        metrics_with_explanations_df = pd.DataFrame(metrics_with_explanations)
        def apply_color(row):
            color, _ = get_color_and_description_overall(row["Metric"], row["Result"])
            return [color if col == "Result" else "" for col in row.index]
        styled_metrics = (
            metrics_with_explanations_df
            .style
            .format({"Result": "{:.4f}"})
            .apply(apply_color, axis=1)
        )
        st.subheader(f"Folder: `{folder}`")
        st.write("**Metrics with Explanations:**")
        st.table(styled_metrics)
else:
    st.write("No folders selected for comparison.")
