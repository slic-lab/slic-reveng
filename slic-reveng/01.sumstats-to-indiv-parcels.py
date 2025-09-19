from scipy.stats import norm
from pathlib import Path
import enigmatoolbox
import pandas as pd
import numpy as np
import os

from statsmodels.sandbox.regression.try_treewalker import inddict
from trimesh.voxel.morphology import surface

# Load the summary statistics file
file_path = "/Users/lars2776/Downloads/tlemtsl_case-controls_CortThick.csv"
summary_stats = pd.read_csv(file_path)


# Define simulation function
def simulate_subject_data(cohen_d, se, n_patients, n_controls, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Estimate pooled standard deviation from SE
    sd_pooled = np.sqrt(((n_patients + n_controls) / (n_patients * n_controls)) * se ** 2)

    # Control group: mean = 0
    control_data = np.random.normal(loc=0, scale=sd_pooled, size=n_controls)

    # Patient group: mean = d * sd_pooled
    patient_mean = cohen_d * sd_pooled
    patient_data = np.random.normal(loc=patient_mean, scale=sd_pooled, size=n_patients)

    return control_data, patient_data


# Extract region names and sizes
region_names = summary_stats['Structure'].tolist()
n_controls = int(summary_stats['n_controls'].median())
n_patients = int(summary_stats['n_patients'].median())
n_regions = len(region_names)

# Initialize matrices
ctrl_matrix = np.zeros((n_controls, n_regions))
pt_matrix = np.zeros((n_patients, n_regions))

# Simulate data per region
for i, row in summary_stats.iterrows():
    d, se = row['d_icv'], row['se_icv']
    ctrl_data, pt_data = simulate_subject_data(d, se, n_patients, n_controls, seed=i)
    ctrl_matrix[:, i] = ctrl_data
    pt_matrix[:, i] = pt_data

# Format as DataFrames with subject IDs and group labels
ctrl_df = pd.DataFrame(ctrl_matrix, columns=region_names)
ctrl_df["group"] = "control"
ctrl_df["subject_id"] = [f"C_{i:04d}" for i in range(n_controls)]

pt_df = pd.DataFrame(pt_matrix, columns=region_names)
pt_df["group"] = "patient"
pt_df["subject_id"] = [f"P_{i:04d}" for i in range(n_patients)]

# Combine both groups into one long DataFrame
full_df = pd.concat([ctrl_df, pt_df], axis=0)

# Reorder columns to have subject ID and group first
cols = ["subject_id", "group"] + region_names
full_df = full_df[cols]

# Display result
print(full_df.head())

# Use brainstat to compare HC to TLE
from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical
from brainstat.stats.terms import FixedEffect
from brainstat.stats.SLM import SLM

d = np.vstack((ctrl_df.iloc[:, :-2].values, pt_df.iloc[:, :-2].values))
grp = np.append(ctrl_df["group"].values, pt_df["group"].values)
grp = np.where(grp == "control", "HC", "PX")
model = FixedEffect(grp, 'grp')
contrast = (grp == "PX").astype(int) - (grp == "HC").astype(int)
slm = SLM(model=model, contrast=contrast, correction="fdr")
slm.fit(d)
plot_cortical(array_name=parcel_to_surface(slm.t.flatten(), 'aparc_conte69'), surface_name="conte69",
                                  size=(2000, 1500), color_bar=True, nan_color=(1, 1, 1, 1), zoom=1.15,
                                  color_range=(-9, 9), cmap='RdBu_r')

# Fetch original summary statistics from ENIGMA toolbox
from enigmatoolbox.datasets import load_summary_stats
sum_stats = load_summary_stats('epilepsy')
CT = sum_stats['CortThick_case_vs_controls_ltle']
CT_d = CT['d_icv']
plot_cortical(array_name=parcel_to_surface(CT_d, 'aparc_conte69'), surface_name="conte69",
              cmap='RdBu_r', color_bar=True, color_range=(-0.5, 0.5), zoom=1.15, size=(2000, 1500))

# Print correlation between simulated t-value maps and sum stats map
print(f"correlation between sum stats and slm.t = {np.corrcoef(CT_d.values, slm.t.flatten())[0, 1]}")