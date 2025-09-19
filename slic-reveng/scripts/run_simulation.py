import pandas as pd
import numpy as np
from slic_reveng.simulate import simulate_subject_data
from slic_reveng.brainstat_analysis import run_glm
from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical
from enigmatoolbox.datasets import load_summary_stats

# Load summary stats
summary_stats = load_summary_stats('epilepsy')['CortThick_case_vs_controls_ltle']
region_names = summary_stats['Structure'].tolist()
n_controls = int(summary_stats['n_controls'].median())
n_patients = int(summary_stats['n_patients'].median())

# Simulate
ctrl_matrix, pt_matrix = [], []
for i, row in summary_stats.iterrows():
    d, se = row['d_icv'], row['se_icv']
    ctrl, pt = simulate_subject_data(d, se, n_patients, n_controls, seed=i)
    ctrl_matrix.append(ctrl)
    pt_matrix.append(pt)

ctrl_df = pd.DataFrame(np.array(ctrl_matrix).T, columns=region_names)
pt_df = pd.DataFrame(np.array(pt_matrix).T, columns=region_names)
ctrl_df["group"] = "control"
pt_df["group"] = "patient"
ctrl_df["subject_id"] = [f"C_{i:04d}" for i in range(n_controls)]
pt_df["subject_id"] = [f"P_{i:04d}" for i in range(n_patients)]

# Run GLM
slm = run_glm(ctrl_df, pt_df)

# Plot simulated vs ENIGMA summary stats
plot_cortical(parcel_to_surface(slm.t.flatten(), 'aparc_conte69'), surface_name="conte69", color_bar=True)
plot_cortical(parcel_to_surface(summary_stats['d_icv'], 'aparc_conte69'), surface_name="conte69", color_bar=True)

# Correlate every subject's maps with the summary statistics map