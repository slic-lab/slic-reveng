import numpy as np

def simulate_subject_data(cohen_d, se, n_patients, n_controls, seed=None):
    if seed is not None:
        np.random.seed(seed)

    sd_pooled = np.sqrt(((n_patients + n_controls) / (n_patients * n_controls)) * se ** 2)

    control_data = np.random.normal(loc=0, scale=sd_pooled, size=n_controls)
    patient_data = np.random.normal(loc=cohen_d * sd_pooled, scale=sd_pooled, size=n_patients)

    return control_data, patient_data