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

# Sidebar option to select metrics type
display_clean = st.sidebar.radio("Select Metrics Type", ["No Clean", "Clean"], key="metrics_type_radio")

def process_folder(folder, display_clean):
    """
    Reads the metrics.csv, metrics_clean.csv, and log.csv files from `folder`.
    Returns:
        - A DataFrame indexed by metric, with a single column named <folder> of results
        - The log data DataFrame (training/val curves)
    If Clean is chosen, merges the 'clean' and 'no_clean' metrics before returning.
    """
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

    # If user wants to combine Clean + No Clean, recalculate aggregate metrics
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

        # Recompute Weighted (by slices) Averages for Dice, IoU
        total_slices = combined_metrics["Total Slices"]
        combined_metrics["Dice"] = (
            no_clean_dict["Dice"] * no_clean_dict["Total Slices"]
            + clean_dict["Dice"] * clean_dict["Total Slices"]
        ) / total_slices
        combined_metrics["IoU"] = (
            no_clean_dict["IoU"] * no_clean_dict["Total Slices"]
            + clean_dict["IoU"] * clean_dict["Total Slices"]
        ) / total_slices

        # Recompute Precision, Recall, FPPS
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

        # Convert dict to DataFrame
        metrics_data = pd.DataFrame(
            [{"Metric": k, "Result": v} for k, v in combined_metrics.items()]
        )
    else:
        # Just use the no_clean data
        metrics_data = metrics_data_no_clean

    # Sort metrics data by our preferred order
    metrics_data["Order"] = metrics_data["Metric"].apply(
        lambda x: METRIC_ORDER.index(x) if x in METRIC_ORDER else len(METRIC_ORDER)
    )
    metrics_data = (
        metrics_data.sort_values("Order")
                    .drop(columns=["Order"])
                    .set_index("Metric")
    )

    # Rename the "Result" column to the folder name
    metrics_data.columns = [folder]
    return metrics_data, log_data


# -------------------------------------------------------------------
# Main Layout
# -------------------------------------------------------------------

if not selected_folders:
    st.title("Explainable XAI for Lung Nodule Segmentation - Overall Results")
    st.write("Please select a folder to view the results.")
    st.stop()

comparison_data = []
logs_data = {}

# Collect the metrics from each folder
for folder in selected_folders:
    folder_data, log_data = process_folder(folder, display_clean)
    if folder_data is not None:
        comparison_data.append(folder_data)
    if log_data is not None:
        logs_data[folder] = log_data

# If we have logs_data, let's show the training/val metric graphs
if logs_data:
    st.subheader("**Training & Validation Curves**")
    graph_metric = st.selectbox("Select Metric to Graph", ["Loss", "IoU", "Dice"])
    show_training = st.checkbox("Show All Training Curves", value=True)
    show_validation = st.checkbox("Show All Validation Curves", value=True)

    # Start a blank Altair layer
    base_chart = alt.Chart()

    layers = []  # We'll accumulate each folder's lines here

    for folder, log in logs_data.items():
        # Our logs typically have columns like: 'loss', 'val_loss', 'iou', 'val_iou', 'dice', 'val_dice', 'epoch'
        # We'll separate training data from validation data
        #   For example, training data => x='epoch', y=log[graph_metric.lower()]
        #   Validation data => x='epoch', y=log[f"val_{graph_metric.lower()}"]

        # Make sure we have the columns
        if graph_metric.lower() not in log.columns and f"val_{graph_metric.lower()}" not in log.columns:
            continue  # skip if neither training nor validation is present

        # Replace the section where you build `folder_chart` in your logs loop:

        folder_training_color = colors_for_folders.get(folder, "#000000")
        folder_validation_color = lighten_color(folder_training_color, factor=0.5)

        chart_data = pd.DataFrame({"epoch": log["epoch"]})
        if show_training and graph_metric.lower() in log.columns:
            chart_data["training"] = log[graph_metric.lower()]
        if show_validation and f"val_{graph_metric.lower()}" in log.columns:
            chart_data["validation"] = log[f"val_{graph_metric.lower()}"]

        # Melt to get rows for training and validation, but we won't
        # directly use the melted table for a single line chart.
        melted_data = chart_data.melt(id_vars="epoch", var_name="type", value_name="value")

        # Instead, build two separate charts (one for training, one for validation),
        # each filtering the melted table.
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

        # Layer them so both lines appear for this folder
        folder_chart = alt.layer(training_chart, validation_chart).properties(title=folder)

        layers.append(folder_chart)


    if layers:
        # Combine all folder charts (layer them horizontally or overlay them)
        # If we want them on the same plot, we do alt.layer(*layers)
        # but that merges the lines into one coordinate system. 
        # Typically we want them on a single chart to compare.
        final_chart = alt.layer(*layers).interactive()

        st.altair_chart(final_chart, use_container_width=True)
    else:
        st.write("No valid training/validation columns found to plot.")


# Now combine final metrics across all folders into a single DataFrame
if comparison_data:
    # Each item in `comparison_data` is a DataFrame with index=Metric, columns=[folder].
    # Concatenate along columns => index=Metric, each folder becomes a separate column.
    final_metrics_df = pd.concat(comparison_data, axis=1)  # shape = (#metrics, #folders)

    # --------------------------------------------------
    # 1) Let user pick a "test result metric" to chart
    # --------------------------------------------------
    st.subheader("**Test Results Graph**")

    test_metrics_options = ["Dice", "IoU", "Precision", "Recall", "FPPS"]
    test_metrics_options = [m for m in test_metrics_options if m in final_metrics_df.index]

    if test_metrics_options:
        selected_test_metric = st.selectbox("Select a Test Metric to Graph", test_metrics_options)
        
        # Extract that row => shape = (n_folders,)
        selected_metric_series = final_metrics_df.loc[selected_test_metric]

        # Convert the chosen metric series to a DataFrame:
        df_scatter = pd.DataFrame({
            "Folder": selected_metric_series.index,
            "Value": selected_metric_series.values
        })

        # Build a custom color scale from the user-picked colors
        domain_folders = list(colors_for_folders.keys())    # e.g. ["NestedUNET1", "NestedUNET2"]
        range_colors  = list(colors_for_folders.values())   # e.g. ["#FF0000", "#00FF00", ...]

        scatter_chart = (
            alt.Chart(df_scatter)
            .mark_circle(size=100)
            .encode(
                # Hide the x-axis labels/ticks via axis=None
                x=alt.X("Folder:N", axis=alt.Axis(title=None, labels=False, ticks=False)),
                y=alt.Y("Value:Q", title=selected_test_metric),
                # Use a custom color scale to map each folder to its picked color
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

    # --------------------------------------------------
    # 2) Show each folders final metrics in a table
    # --------------------------------------------------
    for folder, metrics_df in zip(selected_folders, comparison_data):
        # Add explanation for each metric
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
            # color is a background color like "#ffdddd", or an empty string
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
