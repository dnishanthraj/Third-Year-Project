#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Run script with: python stats_script.py --metric "dice_mean" --csv_files path/to/per_patient_metrics.csv path/to/per_patient_metrics_fpr.csv


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Perform paired t-test (if 2 CSVs) or ANOVA (if >2 CSVs) on per-patient metrics."
    )
    parser.add_argument("--metric", type=str, default="dice_mean",
                        help="The column name of the metric to test (e.g. dice_mean, iou_mean)")
    parser.add_argument("--csv_files", nargs="+", required=True,
                        help="Paths to the CSV files containing per-patient metrics. At least 2 required.")
    return parser.parse_args()

def load_and_prepare_data(csv_path, metric, group_label):
    """
    Reads a CSV and returns a DataFrame with at least columns: patient_id and the metric.
    Adds a column 'group' with the provided label.
    """
    df = pd.read_csv(csv_path)
    if 'patient_id' not in df.columns or metric not in df.columns:
        raise ValueError(f"CSV file {csv_path} must contain 'patient_id' and '{metric}' columns.")
    df = df[['patient_id', metric]].copy()
    df['group'] = group_label
    return df

def run_paired_ttest(df1, df2, metric):
    """
    Merge the two DataFrames on patient_id and run a paired t-test.
    """
    merged = pd.merge(df1, df2, on="patient_id", suffixes=('_group1', '_group2'))
    vals1 = merged[f"{metric}_group1"]
    vals2 = merged[f"{metric}_group2"]
    t_stat, p_val = stats.ttest_rel(vals1, vals2)
    return t_stat, p_val, merged

def run_anova(data_dict):
    """
    Runs one-way ANOVA across groups.
    data_dict: dict where keys are group names and values are arrays of data.
    Returns F-statistic, p-value, and a Tukey HSD summary DataFrame.
    """
    all_values = []
    all_groups = []
    for group_name, values in data_dict.items():
        all_values.extend(values)
        all_groups.extend([group_name] * len(values))
    df_long = pd.DataFrame({"value": all_values, "group": all_groups})
    model = sm.OLS.from_formula("value ~ C(group)", data=df_long).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    F_stat = anova_table["F"][0]
    p_val = anova_table["PR(>F)"][0]
    
    tukey = None
    if p_val < 0.05:
        tukey_results = pairwise_tukeyhsd(endog=df_long["value"], groups=df_long["group"], alpha=0.05)
        tukey = tukey_results.summary()
    
    return F_stat, p_val, df_long, tukey

def main():
    args = parse_arguments()
    metric = args.metric
    csv_files = args.csv_files
    n_groups = len(csv_files)
    
    # Create a dictionary to hold DataFrames (group label based on file name)
    groups = {}
    for path in csv_files:
        label = path.split("/")[-1].split('.')[0]  # use file name (without extension) as group label
        groups[label] = load_and_prepare_data(path, metric, label)
    
    if n_groups == 2:
        # For 2 groups, perform a paired t-test.
        group_names = list(groups.keys())
        print(f"Running paired t-test on groups: {group_names[0]} and {group_names[1]}")
        t_stat, p_val, merged = run_paired_ttest(groups[group_names[0]], groups[group_names[1]], metric)
        print(f"Paired t-test results for metric '{metric}':")
        print(f"  t-statistic = {t_stat:.4f}, p-value = {p_val:.6e}")
        if p_val < 0.05:
            print("The difference between the two groups is statistically significant (p < 0.05).")
        else:
            print("No statistically significant difference was found (p >= 0.05).")
        # Optionally, you can save or print the merged DataFrame for inspection:
        # merged.to_csv("merged_results.csv", index=False)
    elif n_groups > 2:
        # For more than 2 groups, run ANOVA.
        print("Running one-way ANOVA on groups:")
        data_dict = {}
        for label, df in groups.items():
            data_dict[label] = df[metric].values
            print(f"  {label}: {df[metric].values}")
        F_stat, p_val, df_long, tukey = run_anova(data_dict)
        print(f"ANOVA results for metric '{metric}':")
        print(f"  F-statistic = {F_stat:.4f}, p-value = {p_val:.6e}")
        if p_val < 0.05:
            print("The overall difference between groups is statistically significant (p < 0.05).")
            print("\nTukey HSD post hoc results:")
            print(tukey)
        else:
            print("No statistically significant difference was found between groups (p >= 0.05).")
        # Optionally, save the long-format DataFrame for further analysis:
        # df_long.to_csv("anova_long_format.csv", index=False)
    else:
        print("Error: Need at least 2 CSV files (i.e., 2 groups) to perform statistical comparisons.")

if __name__ == "__main__":
    main()
