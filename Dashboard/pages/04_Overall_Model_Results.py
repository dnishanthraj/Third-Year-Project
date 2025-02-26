import streamlit as st
import pandas as pd
import os
import altair as alt
import collections
import re

from components import (
    calculate_precision, calculate_recall, calculate_fpps,
    get_color_and_description_overall, load_log_data, lighten_color, run_statistical_tests
)

st.set_page_config(page_title="Overall Results", layout="wide")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
    /* Force Poppins on every element */
    [data-testid="stAppViewContainer"] *, [data-testid="stSidebar"] * {
        font-family: 'Poppins', sans-serif !important;
    }
    .block-container {
        max-width: 1400px !important;  /* or 1600px, adjust to taste */
        padding: 3rem 2rem !important; /* Adjust as needed for your taste */
    }
                /* Increase vertical spacing between nav items */
    /* (Optional) If you want to reduce vertical margin inside sidebar expanders: */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        margin: 0.1rem 0 !important;
    }        
    # [data-testid="stSidebarNav"] ul {
    #     margin-top: 0.5rem; /* space above the list */
    # }
    [data-testid="stSidebarNav"] ul li {
        margin-bottom: 0.2rem; /* space between items */
    }
    /* Add some padding around each link */
    [data-testid="stSidebarNav"] ul li a {
        padding: 0.3rem 1rem; 
        border-radius: 4px;
    }
            /* Increase spacing before h2 headings in the sidebar */
    [data-testid="stSidebar"] h2 {
        margin-bottom: 1rem !important;
    }      
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------
# PATH SETUP
# ---------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_OUTPUTS_DIR = os.path.join(ROOT_DIR, "Project", "Segmentation", "model_outputs")
# Near the top, after imports, or immediately before you use folder_settings
if "folder_settings" not in st.session_state:
    st.session_state["folder_settings"] = {}

# ---------------------------------------------------
# COLLECT ALL AVAILABLE FOLDERS
# ---------------------------------------------------
available_folders = [
    f for f in os.listdir(MODEL_OUTPUTS_DIR)
    if os.path.isdir(os.path.join(MODEL_OUTPUTS_DIR, f))
]

# (Then, if you want a separate heading for your settings)
st.sidebar.markdown("## Sidebar Settings")

# ---------------------------------------------------
# SIDEBAR: FOLDER SETTINGS
# ---------------------------------------------------
with st.sidebar.expander("Folder Settings", expanded=False):
    st.markdown("Pick one or more folders, then optionally rename them.")
    
    # 1) SELECT FOLDERS
    st.markdown("#### Select Folders to Compare")
    st.markdown("Pick one or more folders:")
    selected_folders = st.multiselect(
        "Folders",
        available_folders,
        default=[available_folders[0]] if available_folders else [],
        label_visibility="collapsed"  # Hide the label
    )
    if not selected_folders:
        st.warning("Please select at least one folder to continue.")
        st.stop()

    # 2) RENAME FOLDERS
    st.markdown("#### Rename Folders")
    if selected_folders:
        selected_folder_for_rename = st.selectbox(
            "Select a folder to rename", 
            selected_folders, 
            key="rename_folder_select"
        )
        new_label = st.text_input(
            f"New label for '{selected_folder_for_rename}':",
            value=selected_folder_for_rename,
            key=f"rename_{selected_folder_for_rename}"
        )
        if 'folder_labels' not in st.session_state:
            st.session_state['folder_labels'] = {}
        st.session_state['folder_labels'][selected_folder_for_rename] = (
            f"{new_label} ({selected_folder_for_rename})"
        )
    else:
        st.warning("No folders available for renaming.")
folder_labels = st.session_state.get('folder_labels', {})


# ---------------------------------------------------
# SIDEBAR: GROUP SETTINGS
# ---------------------------------------------------
with st.sidebar.expander("Group Settings", expanded=False):
    st.markdown("Create groups, then assign each folder to a group for color-coding.")

    # 1) Group Creation: No Group and Additional Groups
    st.markdown("#### Group Creation")
    groups_dict = {}
    # No Group Settings (default group)
    no_group_name = st.text_input("Name for No Group", value="No Group", key="no_group_name")
    no_group_color = st.color_picker("Color for No Group", "#999999", key="color_no_group")
    groups_dict[no_group_name] = no_group_color

    # Additional Groups (displayed inline, no nested expanders)
    st.markdown("#### Additional Groups")
    num_groups = st.number_input(
        "Number of Additional Groups",
        min_value=0,
        value=0,
        step=1,
        help="Enter how many named groups you want to define (excluding the default 'No Group')."
    )
    for i in range(num_groups):
        group_name = st.text_input(f"Name for Group {i+1}", value=f"Group{i+1}", key=f"group_name_{i}")
        default_color = "#008000"
        group_color = st.color_picker(f"Color for Group {i+1}", default_color, key=f"group_color_{i}")
        groups_dict[group_name] = group_color

    # 2) Folder Assignment: Assign each selected folder to a group
    st.markdown("#### Assign Folders to Groups")
    folder_to_group = {}
    for folder in selected_folders:
        display_name = folder_labels.get(folder, folder)
        assigned_group = st.selectbox(
            f"Group for folder: {display_name}",
            list(groups_dict.keys()),
            key=f"group_select_{folder}"
        )
        folder_to_group[folder] = assigned_group


# ---------------------------------------------------
# MAP EACH FOLDER TO ITS GROUP COLOR
# ---------------------------------------------------
colors_for_folders = {
    folder: groups_dict[folder_to_group[folder]]
    for folder in selected_folders
}

# # ---------------------------------------------------
# # EXTRA SIDEBAR: PER-FOLDER METRIC SETTINGS (Single Folder)
# # ---------------------------------------------------
# st.sidebar.markdown("### Choose Group Results")
# st.sidebar.markdown("""Toggle between Clean/Without Clean and Raw/FPR Results for each group.""")

# with st.sidebar.expander("Per-Folder Metric Settings", expanded=False):
#     st.markdown("Select a folder to configure its metric settings:")

#     folder_options = {folder_labels.get(folder, folder): folder for folder in selected_folders}
#     selected_folder_label = st.selectbox("Folder", list(folder_options.keys()))
#     selected_folder = folder_options[selected_folder_label]

#     st.markdown(f"**Configure settings for: {selected_folder_label}**")

#     metric_source = st.radio(
#         "Segmentation Source",
#         ["Raw", "FPR"],
#         key=f"metrics_source_{selected_folder}"
#     )
#     clean_option = st.radio(
#         "Include Clean Set",
#         ["No Clean", "Clean"],
#         key=f"display_clean_{selected_folder}"
#     )

#     # Save to session_state so all folders can have their own config
#     st.session_state["folder_settings"][selected_folder] = {
#         "metrics_source": metric_source,
#         "display_clean": clean_option
#     }





# ---------------------------------------------------
# METRIC ORDER & HELPER: Additional metrics
# ---------------------------------------------------
METRIC_ORDER = [
    "Dice", "IoU", "Accuracy", "Precision", "Recall", "Specificity", "F1-Score",
    "FPPS", "Total Slices", "Total Patients",
    "True Positive (TP)", "True Negative (TN)", "False Positive (FP)", "False Negative (FN)"
]

def compute_additional_metrics(metrics_dict):
    """
    Given a dictionary of basic counts and existing metrics:
      { "Dice": ..., "IoU": ..., "True Positive (TP)": ... etc. }
    compute Accuracy, Specificity, and F1-Score if possible, and store them back.
    """
    TP = metrics_dict.get("True Positive (TP)", 0)
    TN = metrics_dict.get("True Negative (TN)", 0)
    FP = metrics_dict.get("False Positive (FP)", 0)
    FN = metrics_dict.get("False Negative (FN)", 0)
    total = TP + TN + FP + FN

    # Accuracy
    if total > 0:
        accuracy = (TP + TN) / total
        metrics_dict["Accuracy"] = accuracy
    else:
        metrics_dict["Accuracy"] = 0.0

    # Specificity
    denom_sp = TN + FP
    if denom_sp > 0:
        specificity = TN / denom_sp
        metrics_dict["Specificity"] = specificity
    else:
        metrics_dict["Specificity"] = 0.0

    # F1-Score
    precision = metrics_dict.get("Precision", 0)
    recall = metrics_dict.get("Recall", 0)
    denom_f1 = precision + recall
    if denom_f1 > 0:
        f1 = 2 * precision * recall / denom_f1
        metrics_dict["F1-Score"] = f1
    else:
        metrics_dict["F1-Score"] = 0.0

    return metrics_dict

# ---------------------------------------------------
# HELPER FUNCTION: PROCESS_FOLDER
# ---------------------------------------------------
def process_folder(folder, display_clean, metrics_source):
    """
    Loads metrics & log files from the specified folder.
    If 'Clean' is chosen, merges the standard and clean dataset metrics.
    Also computes Accuracy, Specificity, F1-Score on the final combined dict.
    Returns a DataFrame of final metrics, or None if there's a file error,
    plus the log DataFrame if found.
    """
    if metrics_source == "FPR":
        metrics_file_path = os.path.join(MODEL_OUTPUTS_DIR, folder, "metrics", "metrics_fpr.csv")
        metrics_clean_file_path = os.path.join(MODEL_OUTPUTS_DIR, folder, "metrics", "metrics_fpr_clean.csv")
    else:
        metrics_file_path = os.path.join(MODEL_OUTPUTS_DIR, folder, "metrics", "metrics.csv")
        metrics_clean_file_path = os.path.join(MODEL_OUTPUTS_DIR, folder, "metrics", "metrics_clean.csv")

    log_file_path = os.path.join(MODEL_OUTPUTS_DIR, folder, "log.csv")

    # Load CSVs
    try:
        metrics_data_no_clean = pd.read_csv(metrics_file_path)
        metrics_data_clean = pd.read_csv(metrics_clean_file_path)
        log_data = load_log_data(log_file_path)
    except FileNotFoundError:
        # st.warning(f"Metrics or log file not found for '{folder}'.")
        return None, None

    no_clean_dict = metrics_data_no_clean.set_index("Metric").to_dict()["Result"]
    clean_dict = metrics_data_clean.set_index("Metric").to_dict()["Result"]

    if display_clean == "Clean":
        combined = {}
        combined["Total Slices"] = no_clean_dict["Total Slices"] + clean_dict["Total Slices"]
        combined["Total Patients"] = no_clean_dict["Total Patients"] + clean_dict["Total Patients"]

        TP = no_clean_dict["True Positive (TP)"] + clean_dict["True Positive (TP)"]
        TN = no_clean_dict["True Negative (TN)"] + clean_dict["True Negative (TN)"]
        FP = no_clean_dict["False Positive (FP)"] + clean_dict["False Positive (FP)"]
        FN = no_clean_dict["False Negative (FN)"] + clean_dict["False Negative (FN)"]

        combined["True Positive (TP)"] = TP
        combined["True Negative (TN)"] = TN
        combined["False Positive (FP)"] = FP
        combined["False Negative (FN)"] = FN

        total_slices = combined["Total Slices"]
        if total_slices > 0:
            combined["Dice"] = (
                no_clean_dict["Dice"] * no_clean_dict["Total Slices"] +
                clean_dict["Dice"] * clean_dict["Total Slices"]
            ) / total_slices
            combined["IoU"] = (
                no_clean_dict["IoU"] * no_clean_dict["Total Slices"] +
                clean_dict["IoU"] * clean_dict["Total Slices"]
            ) / total_slices
        else:
            combined["Dice"] = 0.0
            combined["IoU"] = 0.0

        combined["Precision"] = calculate_precision(TP, FP)
        combined["Recall"] = calculate_recall(TP, FN)
        combined["FPPS"] = calculate_fpps(FP, combined["Total Patients"])

        # Additional metrics
        combined = compute_additional_metrics(combined)
        metrics_final = pd.DataFrame([{"Metric": k, "Result": v} for k, v in combined.items()])

    else:
        # No Clean
        base_dict = no_clean_dict.copy()
        TP = base_dict.get("True Positive (TP)", 0)
        TN = base_dict.get("True Negative (TN)", 0)
        FP = base_dict.get("False Positive (FP)", 0)
        FN = base_dict.get("False Negative (FN)", 0)
        if "Precision" not in base_dict:
            base_dict["Precision"] = calculate_precision(TP, FP)
        if "Recall" not in base_dict:
            base_dict["Recall"] = calculate_recall(TP, FN)
        if "FPPS" not in base_dict:
            patients = base_dict.get("Total Patients", 0)
            base_dict["FPPS"] = calculate_fpps(FP, patients)
        
        base_dict = compute_additional_metrics(base_dict)
        metrics_final = pd.DataFrame([{"Metric": k, "Result": v} for k, v in base_dict.items()])

    # Sort by METRIC_ORDER
    metrics_final["Order"] = metrics_final["Metric"].apply(
        lambda x: METRIC_ORDER.index(x) if x in METRIC_ORDER else len(METRIC_ORDER)
    )
    metrics_final = (
        metrics_final.sort_values("Order")
                     .drop(columns=["Order"])
                     .set_index("Metric")
    )
    metrics_final.columns = [folder]

    return metrics_final, log_data

# ---------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------
st.title("Overall Model Results")
st.markdown("---")

if not selected_folders:
    st.write("Please select at least one folder from the sidebar to view results.")
    st.stop()

st.markdown("""
**Overview:**  
- Below, you'll see training/validation curves, a summary scatter plot for chosen metrics, 
  and confusion matrices with additional metrics (including Accuracy, Specificity, F1-Score).  
- Folders are grouped as you assigned in the sidebar. Each group has a single color.  
- The **Raw/FPR** and **Clean/No Clean** settings in the sidebar will affect both the scatter chart 
  and the per-folder confusion matrix.
""")

# ---------------------------------------------------
# 1) LOAD LOG FILES (For Training Curves)
# ---------------------------------------------------
logs_data = {}
for folder in selected_folders:
    log_file_path = os.path.join(MODEL_OUTPUTS_DIR, folder, "log.csv")
    try:
        log_data = load_log_data(log_file_path)
        logs_data[folder] = log_data
    except FileNotFoundError:
        logs_data[folder] = pd.DataFrame()
        st.warning(f"Log file not found for '{folder}'.")

st.subheader("Training & Validation Curves")
st.markdown("""
Compare how each folder learned over epochs. 
The same group color is applied to each folder in that group.  
Check/uncheck to reduce clutter if you have many folders.
""")

graph_metric = st.selectbox("Metric to Graph", ["Loss", "IoU", "Dice"])
show_training = st.checkbox("Show Training Curves", value=True)
show_validation = st.checkbox("Show Validation Curves", value=True)

layers = []
for folder, log_df in logs_data.items():
    train_col = graph_metric.lower()
    val_col = f"val_{graph_metric.lower()}"
    if train_col not in log_df.columns and val_col not in log_df.columns:
        continue

    main_color = colors_for_folders.get(folder, "#000000")
    val_color = lighten_color(main_color, factor=0.5)

    # ?? Add a label column for tooltip display
    chart_data = pd.DataFrame({"epoch": log_df["epoch"]})
    chart_data["folder_label"] = folder_labels.get(folder, folder)

    if show_training and train_col in log_df.columns:
        chart_data["training"] = log_df[train_col]
    if show_validation and val_col in log_df.columns:
        chart_data["validation"] = log_df[val_col]

    melted = chart_data.melt(
        id_vars=["epoch", "folder_label"], 
        var_name="type", 
        value_name="value"
    )

    train_line = (
        alt.Chart(melted)
        .transform_filter("datum.type == 'training'")
        .mark_line(point=True)
        .encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y("value:Q", title=graph_metric),
            color=alt.value(main_color),
            tooltip=["epoch", "value", "folder_label"]
        )
    )

    val_line = (
        alt.Chart(melted)
        .transform_filter("datum.type == 'validation'")
        .mark_line(point=True)
        .encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y("value:Q", title=graph_metric),
            color=alt.value(val_color),
            tooltip=["epoch", "value", "folder_label"]
        )
    )

    # Title uses your group name + the renamed label
    group_name = folder_to_group[folder]
    chart_layer = alt.layer(train_line, val_line).properties(
        title=f"{group_name} - {folder_labels.get(folder, folder)}"
    )
    layers.append(chart_layer)

if layers:
    final_chart = alt.layer(*layers).interactive()
    st.altair_chart(final_chart, use_container_width=True)
else:
    st.write("No valid columns found to plot (missing training/validation logs).")

# ---------------------------------------------------
# 2) PROCESS SELECTED FOLDERS -> BUILD ONE DICTIONARY FOR METRICS
# ---------------------------------------------------
comparison_data_dict = {}
for folder in selected_folders:
    # If missing, default to Raw + No Clean
    f_settings = st.session_state["folder_settings"].get(folder, {
        "metrics_source": "Raw",
        "display_clean": "No Clean"
    })
    folder_data, _ = process_folder(
        folder,
        display_clean=f_settings["display_clean"],
        metrics_source=f_settings["metrics_source"]
    )
    comparison_data_dict[folder] = folder_data if folder_data is not None else pd.DataFrame()


# ---------------------------------------------------
# 3) TEST RESULTS BAR CHART (REPLACING SCATTER)
# ---------------------------------------------------
st.subheader("Test Results Graph")
st.markdown("""
Choose which metric to compare across all folders. Each bar represents one folder, colored by the group.
""")

final_metrics_list = []
for folder, metrics_df in comparison_data_dict.items():
    if not metrics_df.empty:
        final_metrics_list.append(metrics_df)

if final_metrics_list:
    final_metrics_df = pd.concat(final_metrics_list, axis=1)
    candidate_metrics = ["Dice", "IoU", "Precision", "Recall", "Specificity", "Accuracy", "F1-Score", "FPPS"]
    candidate_metrics = [m for m in candidate_metrics if m in final_metrics_df.index]

    if candidate_metrics:
        chosen_metric = st.selectbox("Select a Metric for Bar Chart", candidate_metrics)
        if chosen_metric in final_metrics_df.index:
            selected_metric_series = final_metrics_df.loc[chosen_metric]
        else:
            st.warning(f"No data for metric '{chosen_metric}'.")
            selected_metric_series = pd.Series(dtype=float)

        # Prepare DataFrame for the bar chart
        bar_data = []
        for folder_name in selected_metric_series.index:
            val = selected_metric_series[folder_name]
            bar_data.append({
                "Folder": folder_labels.get(folder_name, folder_name),  # Use any custom folder label
                "Value": val,
                "Group": folder_to_group[folder_name],
            })
        bar_df = pd.DataFrame(bar_data)

        # Build a bar chart instead of a scatter chart
        unique_groups = bar_df["Group"].unique().tolist()
        group_color_map = {g: groups_dict[g] for g in unique_groups if g in groups_dict}

        bar_chart = (
            alt.Chart(bar_df)
            .mark_bar()
            .encode(
                x=alt.X("Folder:N", sort=None, title="Folder"),
                y=alt.Y("Value:Q", title=chosen_metric),
                color=alt.Color(
                    "Group:N",
                    scale=alt.Scale(domain=list(group_color_map.keys()),
                                    range=list(group_color_map.values())),
                    legend=alt.Legend(title="Group")
                ),
                tooltip=["Folder:N", "Value:Q", "Group:N"]
            )
            .properties(
                width=600,
                height=400,
                title=f"{chosen_metric} Across Folders"
            )
        )
        st.altair_chart(bar_chart, use_container_width=True)

        # Identify the highest & lowest scoring folders
        if not bar_df.empty:
            # Sort by 'Value'
            sorted_df = bar_df.sort_values(by="Value", ascending=False)
            highest = sorted_df.iloc[0]
            lowest = sorted_df.iloc[-1]

            st.markdown("**Insights:**")
            st.write(f"- **Highest** {chosen_metric}: {highest['Folder']} ({highest['Value']:.4f})")
            st.write(f"- **Lowest** {chosen_metric}: {lowest['Folder']} ({lowest['Value']:.4f})")
            gap = highest["Value"] - lowest["Value"]
            st.write(f"The performance gap between highest and lowest is **{gap:.4f}**, "
                     "suggesting how much variation there is among selected folders.")
    else:
        st.warning("No suitable metrics found for the bar chart.")
else:
    st.write("No folders have any metrics to display in this section.")


# ---------------------------------------------------
# 4) PER-FOLDER METRICS + CONFUSION MATRIX
# ---------------------------------------------------

st.subheader("Folder Metrics & Confusion Matrix")
st.markdown("""
Choose a folder from the dropdown below to see that folder's confusion matrix and other metrics.
""")

if selected_folders:
    label_to_folder = { folder_labels.get(folder, folder): folder for folder in selected_folders }
    chosen_label = st.selectbox("Select a folder to view metrics", list(label_to_folder.keys()))
    chosen_folder = label_to_folder[chosen_label]

    # --- New: Integrate toggle options here ---
    st.markdown("#### Choose Segmentation Source & Dataset Type")
    # Set default values if not yet in session_state
    if f"metrics_source_{chosen_folder}" not in st.session_state:
         st.session_state[f"metrics_source_{chosen_folder}"] = "Raw"
    if f"display_clean_{chosen_folder}" not in st.session_state:
         st.session_state[f"display_clean_{chosen_folder}"] = "No Clean"

    metric_source = st.radio(
         "Segmentation Source",
         ["Raw", "FPR"],
         index=["Raw", "FPR"].index(st.session_state[f"metrics_source_{chosen_folder}"]),
         key=f"metrics_source_{chosen_folder}"
    )
    clean_option = st.radio(
         "Include Clean Set",
         ["No Clean", "Clean"],
         index=["No Clean", "Clean"].index(st.session_state[f"display_clean_{chosen_folder}"]),
         key=f"display_clean_{chosen_folder}"
    )

    # Save the latest settings to session_state
    st.session_state["folder_settings"][chosen_folder] = {
         "metrics_source": metric_source,
         "display_clean": clean_option
    }
    # --- End new toggle integration ---

    # Recompute metrics for the chosen folder using the current toggles
    folder_data, _ = process_folder(chosen_folder, display_clean=clean_option, metrics_source=metric_source)

    with st.expander(f"View Metrics for: {chosen_folder}", expanded=True):
        if folder_data is None or folder_data.empty:
            st.write("No metrics found for this folder.")
        else:
            # --- Confusion Matrix ---
            def safe_val(metric_name):
                if metric_name in folder_data.index and chosen_folder in folder_data.columns:
                    return folder_data.loc[metric_name, chosen_folder]
                else:
                    return 0

            TP = safe_val("True Positive (TP)")
            TN = safe_val("True Negative (TN)")
            FP = safe_val("False Positive (FP)")
            FN = safe_val("False Negative (FN)")

            cm_data = [
                ["", "Predicted Positive", "Predicted Negative"],
                ["Actual Positive", f"TP={TP}", f"FN={FN}"],
                ["Actual Negative", f"FP={FP}", f"TN={TN}"]
            ]
            df_cm = pd.DataFrame(cm_data)

            def style_cm(df):
                color_map = pd.DataFrame("", index=df.index, columns=df.columns)
                cell_map = {
                    (1, 1): ("True Positive (TP)", "TP"),
                    (1, 2): ("False Negative (FN)", "FN"),
                    (2, 1): ("False Positive (FP)", "FP"),
                    (2, 2): ("True Negative (TN)", "TN")
                }
                for (r, c), (metric_name, _) in cell_map.items():
                    cell_val = df.iloc[r, c]
                    match = re.search(r"=(\d+\.?\d*)", cell_val)
                    val = float(match.group(1)) if match else 0.0
                    bg, _ = get_color_and_description_overall(metric_name, val)
                    color_map.iloc[r, c] = f"background-color: {bg}"
                return color_map

            st.write("**Confusion Matrix:**")
            st.table(df_cm.style.apply(style_cm, axis=None))

            # --- Other Metrics Table ---
            confusion_metrics = {"True Positive (TP)", "True Negative (TN)", "False Positive (FP)", "False Negative (FN)"}
            rows_other = []
            for metric in folder_data.index:
                if metric in confusion_metrics:
                    continue
                score = folder_data.loc[metric, chosen_folder]
                bg, desc = get_color_and_description_overall(metric, score)
                rows_other.append({
                    "Metric": metric,
                    "Result": score,
                    "Description": desc
                })

            if rows_other:
                df_other = pd.DataFrame(rows_other)
                def apply_colors(row):
                    c, _ = get_color_and_description_overall(row["Metric"], row["Result"])
                    return [c if col == "Result" else "" for col in row.index]
                st.write("**Other Metrics:**")
                st.table(df_other.style.format({"Result": "{:.4f}"}).apply(apply_colors, axis=1))
            else:
                st.write("No additional metrics to display.")

st.markdown("---")
st.markdown(f"""
**Tip:**  
If you have a large number of groups, keep group expanders collapsed 
to avoid a very long sidebar. Folders can remain in **{no_group_name}** if you don't wish to group them further.
""")
