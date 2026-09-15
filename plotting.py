# Example of making a revenue contour plot
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.interpolate import griddata
import pandas as pd
from matplotlib.colors import SymLogNorm

plt.rcParams.update({'text.usetex':'True', 'font.size':20})
#results = json.load(open('outdir_rdisk_sig_tilt_1/results.json'))
#savename = 'var_sigma_fix_theta_1'
results = json.load(open('outdir_rdisk_theta_sig_20/results.json'))
savename = 'var_theta_fix_sig_20'


r_BBH = np.array([results[key]["r_BBH"] for key in results.keys()])
sigma = np.array([results[key]["sigma"] for key in results.keys()])
tilt = np.array([results[key]["tilt"] for key in results.keys()])
log_B = np.array([results[key]["logB_AGN_GAS"] for key in results.keys()])
err = np.array([results[key]["logB_err"] for key in results.keys()])
ratio = np.array([results[key]["median_R"] for key in results.keys()])

log_B = log_B / 2.303
err = err / 2.303

df = pd.DataFrame({
    "r": r_BBH,
    "tilt": tilt,
    "sigma": sigma,
    "logB": log_B,
    "coeff": ratio
})

table = df.pivot(index="tilt", columns="r", values="logB")
R, T = np.meshgrid(table.columns.values, table.index.values)
Z = table.values

fig, ax = plt.subplots(figsize=(8, 6))
levels = np.linspace(0, 10, 100)

cf = ax.contourf(
    R, T, Z,
    levels=levels,
    cmap="magma",
    extend="both",
    norm=SymLogNorm(linthresh=1)
)

#ax.scatter(
#    r_BBH, tilt,
#    c=log_B,
#    cmap="magma",
#    edgecolors="white",
#    s=20,
#    vmin=0,
#    vmax=10
#)

# Colorbar
cbar = fig.colorbar(cf, ax=ax, ticks=[0, 2, 4, 6, 8, 10])
cbar.set_label(r"$\log_{10} \mathcal{B}^{\rm disk}_{\rm gas}$")
ax.set_xscale('log')
#ax.set_yscale('log')
ax.text(0.75, 0.95, 
        r"$\sigma_{D_L} = 20 \%$",
        color='white',
        transform=ax.transAxes,
        verticalalignment='top',
        size=20)
ax.set_xlabel(r"$r_{\rm BBH} / R_S$")
#ax.set_ylabel(r"$\sigma_{D_L} / D_L$")
#=ax.set_ylabel(r"$\theta$")

plt.tight_layout()
plt.savefig("plots/logB_"+savename+".pdf", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

plt.scatter(r_BBH, ratio, c=log_B, cmap='magma',vmin=0, vmax=10, s=10, marker='d')
plt.xlabel(r"$r_{\rm BBH}$")
plt.ylabel(r"$(1 + z_{\rm grav})(1 + z_{\rm rel})$")
plt.yscale('log')
plt.xscale('log')
plt.savefig("plots/coeff_"+savename+".pdf", dpi=300, bbox_inches='tight')
plt.show()
