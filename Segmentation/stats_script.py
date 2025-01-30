# Run with:

# ./stats_script.py --metric "Dice" \
#     --groups "A=0.52,0.53,0.50" \
#               "B=0.48,0.49,0.51" \
#               "C=0.55,0.56,0.54"


#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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
       - (F_stat, p_val)
       - posthoc_df (DataFrame) if p < 0.05, else None
    """
    # 1) Convert data_dict to a "long" DataFrame => columns=["value", "group"]
    all_values = []
    all_groups = []
    for group_name, values in data_dict.items():
        all_values.extend(values)
        all_groups.extend([group_name] * len(values))

    df_long = pd.DataFrame({"value": all_values, "group": all_groups})

    # 2) Fit OLS model and run ANOVA
    model = ols("value ~ C(group)", data=df_long).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    F_stat = anova_table["F"][0]
    p_val = anova_table["PR(>F)"][0]

    # 3) If ANOVA is significant, run Tukey post-hoc
    posthoc_df = None
    if p_val < 0.05:
        tukey = pairwise_tukeyhsd(endog=df_long["value"], groups=df_long["group"], alpha=0.05)
        posthoc_df = pd.DataFrame(data=tukey.summary()[1:], columns=tukey.summary()[0])

    return (F_stat, p_val), posthoc_df

def main():
    """
    Example usage of the script:

    python stats_script.py --metric "Dice" \
       --groups "A=0.52,0.53,0.50" "B=0.48,0.49,0.51"

    Or for 3+ groups (ANOVA):
    python stats_script.py --metric "Dice" \
       --groups "A=0.52,0.53,0.50" "B=0.48,0.49,0.51" "C=0.55,0.56,0.54"
    """

    parser = argparse.ArgumentParser(
        description="Perform T-Test or ANOVA on multiple groups of numeric data."
    )
    parser.add_argument("--metric", type=str, default="MetricName",
                        help="Name of the metric being tested (for display).")
    parser.add_argument("--groups", nargs="+", required=True,
                        help=(
                            "Group data in the form 'GroupName=val1,val2,val3'. "
                            "Can specify multiple such arguments."
                        ))

    args = parser.parse_args()

    # Parse the --groups inputs into a dictionary
    # For example, "A=0.52,0.53,0.50" => data_dict["A"] = [0.52, 0.53, 0.50]
    data_dict = {}
    for group_arg in args.groups:
        # group_arg looks like "A=0.52,0.53,0.50"
        group_name, values_str = group_arg.split("=")
        # values_str => "0.52,0.53,0.50"
        values = [float(x) for x in values_str.split(",")]
        data_dict[group_name] = values

    n_groups = len(data_dict)

    if n_groups < 2:
        print("Error: Need at least 2 groups to compare.")
        return

    if n_groups == 2:
        # T-Test
        group_names = list(data_dict.keys())
        a_name, b_name = group_names[0], group_names[1]
        a_vals = data_dict[a_name]
        b_vals = data_dict[b_name]

        t_stat, p_val = run_t_test(a_vals, b_vals)
        print(f"Metric: {args.metric}")
        print(f"T-Test comparing {a_name} vs {b_name}")
        print(f"  t-statistic = {t_stat:.4f}, p-value = {p_val:.6e}")

    else:
        # ANOVA + possible Tukey
        (F_stat, p_val), posthoc_df = run_anova(data_dict)
        print(f"Metric: {args.metric}")
        print("ANOVA across groups:", ", ".join(data_dict.keys()))
        print(f"  F-stat = {F_stat:.4f}, p-value = {p_val:.6e}")

        if p_val < 0.05 and posthoc_df is not None:
            print("\nPost-hoc Tukey HSD results:")
            print(posthoc_df)
        else:
            print("No significant pairwise differences based on Tukey (or p >= 0.05).")


if __name__ == "__main__":
    main()
