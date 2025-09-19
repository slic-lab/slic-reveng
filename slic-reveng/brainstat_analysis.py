import numpy as np
from brainstat.stats.terms import FixedEffect
from brainstat.stats.SLM import SLM

def run_glm(ctrl_df, pt_df):
    d = np.vstack((ctrl_df.iloc[:, :-2].values, pt_df.iloc[:, :-2].values))
    grp = np.append(ctrl_df["group"].values, pt_df["group"].values)
    grp = np.where(grp == "control", "HC", "PX")
    model = FixedEffect(grp, 'grp')
    contrast = (grp == "PX").astype(int) - (grp == "HC").astype(int)

    slm = SLM(model=model, contrast=contrast, correction="fdr")
    slm.fit(d)
    return slm