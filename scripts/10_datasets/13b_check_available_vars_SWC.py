from pathlib import Path

import pandas as pd
from diive.core.io.files import load_parquet
from src.paths import data_path

# Load datasets info
infile = data_path("data/outputs/10_datasets/12_datasets_info_parquet.csv")
datasets_df = pd.read_csv(infile)

required_vars = dict(
    NEE_VAR=['NEE_VUT_50', 'NEE_vUT_USTAR50', 'NEE_CUT_50'],  # Used to calculate NEP
    NEE_QC_VAR=['NEE_VUT_50_QC', 'NEE_vUT_USTAR50', 'NEE_CUT_50_QC'],
    LE_VAR='LE_F_MDS',  # Used to calculate ET
    LE_QC_VAR='LE_F_MDS_QC',
    GPP_VAR=['GPP_NT_VUT_50', 'GPP_NT_vUT_USTAR50', 'GPP_NT_CUT_50',
             'GPP_DT_VUT_50', 'GPP_DT_vUT_USTAR50', 'GPP_DT_CUT_50'],
    RECO_VAR=['RECO_NT_VUT_50', 'RECO_NT_vUT_USTAR50', 'RECO_NT_CUT_50',
              'RECO_DT_VUT_50', 'RECO_DT_vUT_USTAR50', 'RECO_DT_CUT_50'],
    SWIN_VAR='SW_IN_F',
    SWIN_QC_VAR='SW_IN_F_QC',
    TA_VAR='TA_F',
    TA_QC_VAR='TA_F_QC',
    VPD_VAR='VPD_F',
    VPD_QC_VAR='VPD_F_QC',
    PREC_VAR='P_F',
    SWC_VAR=['SWC_F_MDS_1', 'SWC_F_MDS_2'],
    SWC_QC_VAR=['SWC_F_MDS_1_QC', 'SWC_F_MDS_2_QC'],
)

# Variables to scan for (by substring match)
SCAN_VARIABLES = ['NEE', 'TA', 'SWC', 'SW_IN', 'VPD']

# Initialize tracking dictionaries for all scan variables
scan_var_counts = {}  # {scan_type: {variable_name: site_count}}
scan_var_nonnull = {}  # {scan_type: {variable_name: [non_null_percentages]}}
scan_var_sites = {}  # {scan_type: {variable_name: [site_names]}}

for scan_type in SCAN_VARIABLES:
    scan_var_counts[scan_type] = {}
    scan_var_nonnull[scan_type] = {}
    scan_var_sites[scan_type] = {}

_datasets_df = datasets_df.copy()
for ix, site in _datasets_df.iterrows():

    # --- TODO testing
    # if site['SITE'] != 'CN-Din':
    #     continue
    # if ix > 10:
    #     break
    # --- TODO testing

    print(f"\nLoading data for site #{ix + 1} {site['SITE']} ...")
    filepath = site['_FILEPATH_PARQUET']
    sitedata = load_parquet(filepath)
    available_vars = sitedata.columns

    for var_key, expected_col_name in required_vars.items():
        if isinstance(expected_col_name, str):
            if expected_col_name not in available_vars:
                expected_col_name = '-MISSING-'
                # raise Exception(f"Required variable '{expected_col_name}' not found in site data for site #{ix} {row['SITE']}.")
        elif isinstance(expected_col_name, list):
            found_one = False
            for alt_name in expected_col_name:
                if alt_name in available_vars:
                    if len(sitedata[alt_name].dropna()) > 0:
                        found_one = True
                        expected_col_name = alt_name
                        break
                    else:
                        found_one = False
            if not found_one:
                expected_col_name = '-MISSING-'
                # raise Exception(f"Required variable '{expected_col_name}' not found in site data for site #{ix} {row['SITE']}.")

        datasets_df.loc[ix, var_key] = expected_col_name

    # --- VARIABLE SCANNING FOR MULTIPLE TYPES ---
    # Find all variables matching each scan type and track coverage
    for scan_type in SCAN_VARIABLES:
        vars_in_site = [col for col in available_vars if scan_type in col]

        for var in vars_in_site:
            n_valid = len(sitedata[var].dropna())
            n_total = len(sitedata)
            pct_nonnull = (n_valid / n_total * 100) if n_total > 0 else 0

            # Initialize tracking if this is the first time seeing this variable
            if var not in scan_var_counts[scan_type]:
                scan_var_counts[scan_type][var] = 0
                scan_var_nonnull[scan_type][var] = []
                scan_var_sites[scan_type][var] = []

            # Only count as "found" if it has non-null values
            if n_valid > 0:
                scan_var_counts[scan_type][var] += 1
                scan_var_nonnull[scan_type][var].append(pct_nonnull)
                scan_var_sites[scan_type][var].append(site['SITE'])

# Save to file
datasets_df = datasets_df.reset_index(drop=True)
datasets_df = datasets_df.sort_values(by=['SITE'], inplace=False)
outfile = data_path("data/outputs/10_datasets/13_datasets_info_parquet_vars.csv")
print(f"\n{'-' * 80}\nSaving info about {len(datasets_df)} datasets to file {outfile}.\n{'-' * 80}")
datasets_df.to_csv(outfile, index=False)

# --- GENERATE COVERAGE SUMMARY FOR ALL VARIABLES ---
print(f"\n{'=' * 100}")
print("VARIABLE COVERAGE SUMMARY (NEE, TA, SWC, SW_IN, VPD)")
print(f"{'=' * 100}\n")

# Store all summaries for CSV export
all_summaries = []

# Process each scan variable type
for scan_type in SCAN_VARIABLES:
    print(f"\n{'-' * 100}")
    print(f"{scan_type} VARIABLES")
    print(f"{'-' * 100}")

    # Sort by frequency (descending)
    sorted_vars = sorted(scan_var_counts[scan_type].items(), key=lambda x: x[1], reverse=True)

    if not sorted_vars:
        print(f"No {scan_type} variables found.")
        continue

    # Create summary dataframe for this type
    summary_data = []
    for var, count in sorted_vars:
        pct_coverage = (count / len(datasets_df)) * 100
        avg_nonnull = pd.Series(scan_var_nonnull[scan_type][var]).mean()

        summary_data.append({
            'VARIABLE_TYPE': scan_type,
            'VARIABLE_NAME': var,
            'SITES_WITH_DATA': count,
            'TOTAL_SITES': len(datasets_df),
            'COVERAGE_PERCENT': f"{pct_coverage:.1f}%",
            'AVG_NONNULL_PERCENT': f"{avg_nonnull:.1f}%",
        })
        all_summaries.append(summary_data[-1])

    summary_df = pd.DataFrame(summary_data)

    # Print summary table
    print(f"{'Variable':<30} {'Sites':<10} {'Coverage':<12} {'Avg Non-Null %':<15}")
    print("-" * 100)
    for _, row in summary_df.iterrows():
        print(f"{row['VARIABLE_NAME']:<30} {row['SITES_WITH_DATA']:<10} {row['COVERAGE_PERCENT']:<12} {row['AVG_NONNULL_PERCENT']:<15}")

print(f"\n{'=' * 100}")

# Save all summaries to CSV
summary_outfile = data_path("data/outputs/10_datasets/13b_variables_coverage_summary.csv")
all_summary_df = pd.DataFrame(all_summaries)
all_summary_df.to_csv(summary_outfile, index=False)
print(f"\nCoverage summary saved to: {summary_outfile}\n")

# Print recommendations for each variable type
print(f"{'=' * 100}")
print("RECOMMENDATIONS")
print(f"{'=' * 100}\n")

for scan_type in SCAN_VARIABLES:
    sorted_vars = sorted(scan_var_counts[scan_type].items(), key=lambda x: x[1], reverse=True)
    if sorted_vars:
        best_var = sorted_vars[0]
        pct = (best_var[1] / len(datasets_df)) * 100
        print(f"{scan_type}:")
        print(f"  Best coverage: {best_var[0]} ({best_var[1]} sites, {pct:.1f}%)")
        if len(sorted_vars) > 1:
            for i, (var, count) in enumerate(sorted_vars[1:4], 2):  # Show top 4
                pct = (count / len(datasets_df)) * 100
                print(f"  #{i}: {var} ({count} sites, {pct:.1f}%)")
        print()
    else:
        print(f"{scan_type}: No variables found.\n")
