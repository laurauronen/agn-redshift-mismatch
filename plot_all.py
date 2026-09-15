# Example of making a revenue contour plot
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.interpolate import griddata
import pandas as pd
from matplotlib.colors import SymLogNorm
import seaborn as sns

plt.rcParams.update({'text.usetex':'True', 'font.size':20})
#results = json.load(open('outdir_rdisk_sig_tilt_1/results.json'))
#savename = 'var_sigma_fix_theta_1'
results = json.load(open('outdir_rdisk_all/results.json'))
savename = 'var_all'


r_BBH = np.array([results[key]["r_BBH"] for key in results.keys()])
sigma = np.array([results[key]["sigma"] for key in results.keys()])
tilt = np.array([results[key]["tilt"] for key in results.keys()])
log_B = np.array([results[key]["logB_AGN_GAS"] for key in results.keys()])
err = np.array([results[key]["logB_err"] for key in results.keys()])
ratio = np.array([results[key]["median_R"] for key in results.keys()])

log_B = log_B / 2.303
err = err / 2.303

df = pd.DataFrame({
    r"$r_{\rm BBH}$": r_BBH,
    r"$|\cos\theta|$": np.abs(np.cos(tilt)),
    r"$\sigma_{D_L}$": sigma,
    r"$\Delta{\rm C}$": np.abs(1 - ratio),
    r"$\log_{10}\mathcal{B}$": log_B,
})

corr = df.corr(numeric_only=True)

plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap='magma_r', vmin=-0.5, vmax=0.5,
            annot=True, fmt=r".2f")
plt.tight_layout()
plt.savefig("plots/heatmap_"+savename+".pdf", dpi=300, bbox_inches='tight')
plt.show()
plt.close()
