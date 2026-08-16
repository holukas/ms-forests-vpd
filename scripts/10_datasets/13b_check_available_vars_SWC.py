from pathlib import Path

import pandas as pd
from diive.core.io.files import load_parquet
from src.common import peak_season_months
from src.paths import data_path, resolve_stored_path

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

# Sites that pass the selection criteria in 15_remove_sites.py. That script runs
# after this one, so the file comes from an earlier pass and can be missing. Without
# it, coverage is reported for the scanned sites only.
usedsites_file = data_path("data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv")
if usedsites_file.is_file():
    used_sites = set(pd.read_csv(usedsites_file)['SITE'])
    print(f"Read {len(used_sites)} used sites from {usedsites_file.name}.")
else:
    used_sites = set()
    print(f"{usedsites_file.name} not found. Used-site columns will be empty.")

# Initialize tracking dictionaries for all scan variables
scan_var_counts = {}  # {scan_type: {variable_name: site_count}}
scan_var_nonnull = {}  # {scan_type: {variable_name: [non_null_percentages]}}
scan_var_sites = {}  # {scan_type: {variable_name: [site_names]}}

for scan_type in SCAN_VARIABLES:
    scan_var_counts[scan_type] = {}
    scan_var_nonnull[scan_type] = {}
    scan_var_sites[scan_type] = {}

# One row per site and variable, written to 13b_variables_per_site.csv
per_site_rows = []

_datasets_df = datasets_df.copy()
for ix, site in _datasets_df.iterrows():

    # --- TODO testing
    # if site['SITE'] != 'CN-Din':
    #     continue
    # if ix > 10:
    #     break
    # --- TODO testing

    print(f"\nLoading data for site #{ix + 1} {site['SITE']} ...")
    filepath = resolve_stored_path(site['_FILEPATH_PARQUET'])
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

    # The months the analysis actually uses. A variable can look well covered over
    # the whole record and still be missing when it matters, which is the question
    # for the deeper soil moisture layers. Same definition as the subsets in stage 21.
    gpp_var = datasets_df.loc[ix, 'GPP_VAR']
    if gpp_var != '-MISSING-':
        peak_months = peak_season_months(sitedata, gpp_col=gpp_var)
        in_peak = sitedata.index.month.isin(peak_months)
    else:
        peak_months = []
        in_peak = None

    # --- VARIABLE SCANNING FOR MULTIPLE TYPES ---
    # Find all variables matching each scan type and track coverage
    for scan_type in SCAN_VARIABLES:
        vars_in_site = [col for col in available_vars if scan_type in col]

        for var in vars_in_site:
            n_valid = len(sitedata[var].dropna())
            n_total = len(sitedata)
            pct_nonnull = (n_valid / n_total * 100) if n_total > 0 else 0

            if in_peak is not None:
                peak_data = sitedata.loc[in_peak, var]
                n_total_peak = len(peak_data)
                n_valid_peak = len(peak_data.dropna())
                pct_nonnull_peak = (n_valid_peak / n_total_peak * 100) if n_total_peak > 0 else 0
            else:
                n_total_peak = 0
                n_valid_peak = 0
                pct_nonnull_peak = float('nan')

            per_site_rows.append({
                'SITE': site['SITE'],
                'IGBP': site.get('IGBP'),
                'USED_SITE': site['SITE'] in used_sites,
                'VARIABLE_TYPE': scan_type,
                'VARIABLE_NAME': var,
                'N_VALID': n_valid,
                'N_TOTAL': n_total,
                'PCT_NONNULL': pct_nonnull,
                'N_VALID_PEAK': n_valid_peak,
                'N_TOTAL_PEAK': n_total_peak,
                'PCT_NONNULL_PEAK': pct_nonnull_peak,
                'PEAK_MONTHS': '+'.join(str(m) for m in peak_months),
            })

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

# --- SAVE ONE ROW PER SITE AND VARIABLE ---
# This is the table needed to decide which sites can carry a deeper soil moisture
# layer, and it keeps the site names that the aggregate summary throws away.
per_site_df = pd.DataFrame(per_site_rows)

# What a site would lose by swapping the shallowest soil moisture layer for a deeper
# one. 1.0 means the deeper layer has data wherever layer 1 does, during the season.
# This is the number that decides whether a site can appear in both the shallow and
# the deep run, which the paired comparison needs.
is_swc_layer = per_site_df['VARIABLE_NAME'].str.fullmatch(r'SWC_F_MDS_\d')
layer1_peak = (per_site_df.loc[per_site_df['VARIABLE_NAME'] == 'SWC_F_MDS_1']
               .drop_duplicates('SITE')
               .set_index('SITE')['N_VALID_PEAK'])
reference = per_site_df.loc[is_swc_layer, 'SITE'].map(layer1_peak)
per_site_df['PEAK_VS_SWC1'] = float('nan')
per_site_df.loc[is_swc_layer, 'PEAK_VS_SWC1'] = (
        per_site_df.loc[is_swc_layer, 'N_VALID_PEAK'] / reference.where(reference > 0)
)

per_site_outfile = data_path("data/outputs/10_datasets/13b_variables_per_site.csv")
per_site_df.to_csv(per_site_outfile, index=False)
print(f"\nPer-site variable table saved to: {per_site_outfile}")

n_used_scanned = per_site_df['SITE'][per_site_df['USED_SITE']].nunique()

# --- GENERATE COVERAGE SUMMARY FOR ALL VARIABLES ---
print(f"\n{'=' * 100}")
print("VARIABLE COVERAGE SUMMARY (NEE, TA, SWC, SW_IN, VPD)")
print(f"Scanned sites: {len(datasets_df)}, of which used: {n_used_scanned}")
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

        # Same counts, restricted to the sites the analysis keeps. These are the
        # numbers to quote, because coverage over all scanned sites overstates what
        # is available where it is needed.
        sites_with_data = scan_var_sites[scan_type][var]
        count_used = sum(1 for s in sites_with_data if s in used_sites)
        pct_coverage_used = (count_used / n_used_scanned * 100) if n_used_scanned else float('nan')

        # Coverage inside the four peak-GPP months, used sites only
        peak_rows = per_site_df.loc[
            (per_site_df['VARIABLE_NAME'] == var)
            & (per_site_df['USED_SITE'])
            & (per_site_df['N_VALID'] > 0)
            ]
        # Median, because coverage percentages are skewed by a few near-empty sites
        median_nonnull_peak = peak_rows['PCT_NONNULL_PEAK'].median()

        summary_data.append({
            'VARIABLE_TYPE': scan_type,
            'VARIABLE_NAME': var,
            'SITES_WITH_DATA': count,
            'TOTAL_SITES': len(datasets_df),
            'COVERAGE_PERCENT': f"{pct_coverage:.1f}%",
            'AVG_NONNULL_PERCENT': f"{avg_nonnull:.1f}%",
            'SITES_WITH_DATA_USED': count_used,
            'TOTAL_SITES_USED': n_used_scanned,
            'COVERAGE_PERCENT_USED': f"{pct_coverage_used:.1f}%",
            'MEDIAN_NONNULL_PEAK_PERCENT_USED': f"{median_nonnull_peak:.1f}%",
        })
        all_summaries.append(summary_data[-1])

    summary_df = pd.DataFrame(summary_data)

    # Print summary table
    print(f"{'Variable':<30} {'Sites':<10} {'Coverage':<12} {'Avg Non-Null %':<15} "
          f"{'Used sites':<12} {'Used cov.':<12} {'Peak non-null %':<16}")
    print("-" * 120)
    for _, row in summary_df.iterrows():
        print(f"{row['VARIABLE_NAME']:<30} {row['SITES_WITH_DATA']:<10} {row['COVERAGE_PERCENT']:<12} "
              f"{row['AVG_NONNULL_PERCENT']:<15} {row['SITES_WITH_DATA_USED']:<12} "
              f"{row['COVERAGE_PERCENT_USED']:<12} {row['MEDIAN_NONNULL_PEAK_PERCENT_USED']:<16}")

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

# --- SOIL MOISTURE LAYERS, USED SITES ONLY ---
# This is the answer to whether a deeper-layer run is worth doing: how many of the
# sites in the analysis carry a second or third layer, and whether that layer has
# data during the months the analysis uses.
print(f"{'=' * 100}")
print("SOIL MOISTURE LAYERS ACROSS USED SITES")
print(f"{'=' * 100}\n")

swc_layers = per_site_df.loc[
    per_site_df['VARIABLE_NAME'].str.fullmatch(r'SWC_F_MDS_\d')
    & (per_site_df['USED_SITE'])
    & (per_site_df['N_VALID'] > 0)
    ]

if swc_layers.empty:
    print("No SWC_F_MDS layers found for used sites.\n")
else:
    print(f"{'Layer':<16} {'Sites':<8} {'of used':<10} {'Median peak non-null %':<24} "
          f"{'Median vs SWC_1':<16} {'Sites >=0.9':<12}")
    print("-" * 110)
    for layer, group in swc_layers.groupby('VARIABLE_NAME'):
        n = group['SITE'].nunique()
        pct = (n / n_used_scanned * 100) if n_used_scanned else float('nan')
        ratio = group['PEAK_VS_SWC1']
        n_swappable = int((ratio >= 0.9).sum())
        print(f"{layer:<16} {n:<8} {pct:>6.1f}%    {group['PCT_NONNULL_PEAK'].median():>10.1f}"
              f"              {ratio.median():>8.2f}         {n_swappable:<12}")

    # How many used sites have a deeper layer available at all
    per_site_layers = swc_layers.groupby('SITE')['VARIABLE_NAME'].apply(set)
    has_deeper = sum(1 for layers in per_site_layers if layers - {'SWC_F_MDS_1'})
    print(f"\nUsed sites with any layer below SWC_F_MDS_1: {has_deeper} of {n_used_scanned}")
    print()
