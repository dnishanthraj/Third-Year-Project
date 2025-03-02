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
        description="Perform paired t-test on per-patient metrics (dice_mean and iou_mean) from two CSV files."
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
    
    # Use the parent folder name plus file name to create unique labels for each group.
    group_labels = []
    for path in csv_files:
        # Use the grandparent folder (model folder) plus file name.
        model_folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
        file_name = os.path.basename(path).split('.')[0]
        label = f"{model_folder}_{file_name}"
        group_labels.append(label)
    
    df_A = load_data(csv_files[0], group_labels[0])
    df_B = load_data(csv_files[1], group_labels[1])
    
    # Print average metrics for each group.
    print("Average metrics per group:")
    for label, df in [(group_labels[0], df_A), (group_labels[1], df_B)]:
        dice_mean = df['dice_mean'].mean()
        iou_mean  = df['iou_mean'].mean()
        dice_std = df['dice_mean'].std()
        iou_std = df['iou_mean'].std()
        print(f"  {label} - Dice: {dice_mean:.4f} (std: {dice_std:.4f}), IoU: {iou_mean:.4f} (std: {iou_std:.4f})")
    
    # Perform paired t-test for dice_mean.
    t_stat_dice, p_val_dice, merged_dice = paired_ttest(df_A, df_B, "dice_mean")
    print("\nPaired t-test results for Dice:")
    print(f"  t-statistic = {t_stat_dice:.4f}, p-value = {p_val_dice:.6e}")
    
    # Perform paired t-test for iou_mean.
    t_stat_iou, p_val_iou, merged_iou = paired_ttest(df_A, df_B, "iou_mean")
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

if __name__ == "__main__":
    main()
