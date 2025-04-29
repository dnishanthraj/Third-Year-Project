#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import scipy.stats as stats
import os

# Run script with: 
# python stats_script.py --csv_files path/to/runA_per_patient_metrics.csv path/to/runB_per_patient_metrics.csv

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Perform paired t-test on per-patient metrics (dice_mean and iou_mean) from two CSV files and compute Precision/Recall from metrics files."
    )
    parser.add_argument("--csv_files", nargs="+", required=True,
                        help="Paths to the CSV files containing per-patient metrics. Exactly 2 required.")
    return parser.parse_args()

def load_data(csv_path, group_label):
    """
    Reads a CSV and returns a DataFrame with columns: patient_id, dice_mean, iou_mean.
    Also adds a column 'group' with the provided label.
    """
    df = pd.read_csv(csv_path)
    required_columns = {'patient_id', 'dice_mean', 'iou_mean'}
    if not required_columns.issubset(set(df.columns)):
        raise ValueError(f"CSV file {csv_path} must contain columns: {required_columns}.")
    df = df[['patient_id', 'dice_mean', 'iou_mean']].copy()
    df['group'] = group_label
    return df

def load_metrics(metrics_path):
    """
    Loads Precision, Recall, FPPS, TP, FP, FN, and Total Patients from a metrics CSV file.
    Returns a dictionary with summed values.
    """
    df = pd.read_csv(metrics_path, index_col=0, header=None, names=["Metric", "Result"])
    metrics = df.to_dict()["Result"]

    # Convert extracted values to floats
    TP = float(metrics.get("True Positive (TP)", 0.0))
    FP = float(metrics.get("False Positive (FP)", 0.0))
    FN = float(metrics.get("False Negative (FN)", 0.0))
    total_slices = float(metrics.get("Total Slices", 1.0))  # Default to 1 to avoid division by zero
    total_patients = float(metrics.get("Total Patients", 1.0))  # Default to 1 to avoid division by zero

    # Compute Precision, Recall, and FPPS safely
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    # fpps = FP / total_patients if total_patients > 0 else 0.0  # FPPS per patient
    fpps = FP / total_slices if total_slices > 0 else 0.0


    return {"TP": TP, "FP": FP, "FN": FN, "Total Patients": total_patients, "Total Slices": total_slices, "Precision": precision, "Recall": recall, "FPPS": fpps}


def compute_combined_metrics(metrics_files):
    """
    Sums TP, FP, FN, Total Patients from multiple files and calculates the overall Precision, Recall, and FPPS.
    """
    total_TP = total_FP = total_FN = total_patients = total_slices = 0.0
    
    for metrics_file in metrics_files:
        metrics = load_metrics(metrics_file)
        total_TP += metrics["TP"]
        total_FP += metrics["FP"]
        total_FN += metrics["FN"]
        total_patients += metrics["Total Patients"]
        total_slices += metrics["Total Slices"]
        

    # Compute combined Precision, Recall, and FPPS safely
    combined_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    combined_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    combined_fpps = total_FP / total_slices if total_slices > 0 else 0.0

    return combined_precision, combined_recall, combined_fpps


def paired_ttest(df1, df2, metric):
    """
    Merge two DataFrames on patient_id and perform a paired t-test on the given metric.
    Returns t-statistic, p-value, and the merged DataFrame.
    """
    merged = pd.merge(df1, df2, on="patient_id", suffixes=('_A', '_B'))
    vals_A = merged[f"{metric}_A"]
    vals_B = merged[f"{metric}_B"]
    t_stat, p_val = stats.ttest_rel(vals_A, vals_B)
    return t_stat, p_val, merged

def main():
    args = parse_arguments()
    csv_files = args.csv_files
    if len(csv_files) != 2:
        print("Error: Exactly 2 CSV files are required for a paired t-test.")
        return
    
    # Generate group labels
    group_labels = []
    for path in csv_files:
        model_folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
        file_name = os.path.basename(path).split('.')[0]
        label = f"{model_folder}_{file_name}"
        group_labels.append(label)
    
    # Load per-patient metrics
    df_A = load_data(csv_files[0], group_labels[0])
    df_B = load_data(csv_files[1], group_labels[1])
    
    # Compute per-patient means and standard deviations
    print("Average metrics per group:")
    for label, df in [(group_labels[0], df_A), (group_labels[1], df_B)]:
        dice_mean = df['dice_mean'].mean()
        iou_mean  = df['iou_mean'].mean()
        dice_std = df['dice_mean'].std()
        iou_std = df['iou_mean'].std()
        print(f"  {label} - Dice: {dice_mean:.4f} (std: {dice_std:.4f}), IoU: {iou_mean:.4f} (std: {iou_std:.4f})")

    # Perform paired t-test
    t_stat_dice, p_val_dice, _ = paired_ttest(df_A, df_B, "dice_mean")
    t_stat_iou, p_val_iou, _ = paired_ttest(df_A, df_B, "iou_mean")

    print("\nPaired t-test results for Dice:")
    print(f"  t-statistic = {t_stat_dice:.4f}, p-value = {p_val_dice:.6e}")

    print("\nPaired t-test results for IoU:")
    print(f"  t-statistic = {t_stat_iou:.4f}, p-value = {p_val_iou:.6e}")

    if p_val_dice < 0.05:
        print("\nThe difference in Dice between the two runs is statistically significant (p < 0.05).")
    else:
        print("\nNo statistically significant difference in Dice was found (p >= 0.05).")
    
    if p_val_iou < 0.05:
        print("The difference in IoU between the two runs is statistically significant (p < 0.05).")
    else:
        print("No statistically significant difference in IoU was found (p >= 0.05).")

    # Load Precision, Recall & FPPS from metric files
    base_metric_files = [
        csv_files[0].replace("per_patient_metrics.csv", "metrics.csv"),
        csv_files[0].replace("per_patient_metrics.csv", "metrics_clean.csv"),
    ]

    fpr_metric_files = [
        csv_files[1].replace("per_patient_metrics_fpr.csv", "metrics_fpr.csv"),
        csv_files[1].replace("per_patient_metrics_fpr.csv", "metrics_fpr_clean.csv"),
    ]

    # Compute combined metrics
    base_precision, base_recall, base_fpps = compute_combined_metrics(base_metric_files)
    fpr_precision, fpr_recall, fpr_fpps = compute_combined_metrics(fpr_metric_files)

    # Display Precision, Recall & FPPS
    print("\nPrecision, Recall & FPPS Results:")
    print(f"  {group_labels[0]} - Precision: {base_precision:.4f}, Recall: {base_recall:.4f}, FPPS (Per Patient): {base_fpps:.4f}")
    print(f"  {group_labels[1]} - Precision: {fpr_precision:.4f}, Recall: {fpr_recall:.4f}, FPPS (Per Patient): {fpr_fpps:.4f}")


if __name__ == "__main__":
    main()
