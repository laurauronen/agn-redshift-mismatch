import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.interpolate import griddata
import pandas as pd
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D

def combine_close_values(arr, threshold=5):
    arr = np.sort(arr)
    groups = []
    current = [arr[0]]

    for value in arr[1:]:
        if value > 100:
            if value - current[-1] <= threshold: 
                current.append(value)
            else:
                groups.append(current)
                current = [value]
        else:
            groups.append(current)
            current = [value]
    groups.append(current)

    return np.array([np.mean(group) for group in groups])


plt.rcParams.update({'text.usetex':'True', 'font.size':25})
#results = json.load(open('outdir_rdisk_sig_tilt_1/results.json')) 
#savename = 'var_sigma_fix_theta_1' 

results1 = json.load(open('outdir_rdisk_theta_sig_1/results.json'))
results5 = json.load(open('outdir_rdisk_theta_sig_5/results.json'))
results10 = json.load(open('outdir_rdisk_theta_sig_10/results.json'))
results20 = json.load(open('outdir_rdisk_theta_sig_20/results.json'))
savename = ''

m20 = json.load(open('20.json'))
m50 = json.load(open('50.json'))
m70 = json.load(open('70.json'))

r_BBH = np.array([results1[key]["r_BBH"] for key in results1.keys()])
sigma = np.array([results1[key]["sigma"] for key in results1.keys()])
tilt = np.array([results1[key]["tilt"] for key in results1.keys()])
log_B_1 = np.array([results1[key]["logB_AGN_GAS"] for key in results1.keys()])
log_B_5 = np.array([results5[key]["logB_AGN_GAS"] for key in results5.keys()])
log_B_10 = np.array([results10[key]["logB_AGN_GAS"] for key in results10.keys()])
log_B_20 = np.array([results20[key]["logB_AGN_GAS"] for key in results20.keys()])
#err = np.array([results[key]["logB_err"] for key in results.keys()]) 
#ratio = np.array([results[key]["median_R"] for key in results.keys()]) 

tr = 'Trap radius [$R_S$]'
m2 = np.array([m20[tr][key] for key in m20[tr].keys()])
m5 = np.array([m50[tr][key] for key in m20[tr].keys()])
m7 = np.array([m70[tr][key] for key in m20[tr].keys()])
add = np.array([7, 24.5, 331])
ms = np.concatenate([m2, m5, m7, add])
ms = combine_close_values(ms, 5)

log_B_1 = log_B_1 / 2.303
log_B_5 = log_B_5 / 2.303
log_B_10 = log_B_10 / 2.303
log_B_20 = log_B_20 / 2.303
#err = err / 2.303 

df_1 = pd.DataFrame({ "r": r_BBH, "tilt": tilt, "sigma": sigma, "logB": log_B_1, })
df_5 = pd.DataFrame({ "r": r_BBH, "tilt": tilt, "sigma": sigma, "logB": log_B_5, })
df_10 = pd.DataFrame({ "r": r_BBH, "tilt": tilt, "sigma": sigma, "logB": log_B_10, })
df_20 = pd.DataFrame({ "r": r_BBH, "tilt": tilt, "sigma": sigma, "logB": log_B_20, })

fig, ax = plt.subplots(figsize=(12, 8))
levels = np.asarray([2])
colors = ["bisque", "orange", "mediumvioletred", "darkmagenta"]
        #"darkmagenta","mediumvioletred",
        #  "gold","bisque"]


        #["#20B2A6","#087F83",
        #"#B52B62","#E34B87"]

contours = []

for df, color in zip(
    [df_1, df_5, df_10, df_20],
    colors):

    detection_threshold = 2
    table = df.pivot(index="tilt", columns="r", values="logB")
    R, T = np.meshgrid(table.columns.values, table.index.values)
    Z = table.values

    #cf = ax.contourf(
    #    R, T, Z,
    #    levels=levels,
    #    colors=color,
    #    #linestyles=['--', '-'],
    #    extend="both",
    #    norm=SymLogNorm(linthresh=1)
    #)
    ax.contourf(
        R, T, Z,
        levels=[detection_threshold, np.nanmax(Z)],
        colors=[color],
        alpha=0.9
    )

    cf = ax.contour(
        R, T, Z,
        levels=[detection_threshold],
        colors=color,
        linewidths=2
    )

    contours.append(cf)

label_style = dict(
    fontsize=25,
    fontweight="bold",
    ha="center",
    va="center",
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="none",
        alpha=0.75,
    ),
)

ax.text(3.8, 3, r"${\rm LVK}$", color='k', #colors[3], 
        **label_style)
ax.text(7, 2.7, r"${\rm LVK~(optimistic)}$", color='k', #colors[2], 
        **label_style)
ax.text(9, 2.4, r"${\rm XG}$", color='k', #colors[1], 
        **label_style)
ax.text(60, 2.4, r"${\rm XG~(optimistic)}$", color='k', #colors[0], 
        **label_style)

legend_lines = [
    # sigma
    Line2D([0], [0], color=colors[0], lw=2, label=r"$\sigma=1\%$"),
    Line2D([0], [0], color=colors[1], lw=2, label=r"$\sigma=5\%$"),
    Line2D([0], [0], color=colors[2], lw=2, label=r"$\sigma=10\%$"),
    Line2D([0], [0], color=colors[3], lw=2, label=r"$\sigma=20\%$"),]

#ax.legend(handles=legend_lines, loc="upper right")
# Colorbar 
ax.set_xscale('log')
ax.set_xlabel(r"$r_{\rm BBH} / R_S$")
ax.set_ylabel(r"$\theta$")
ax.vlines(ms, ymin=-3.5, ymax=3.5, color='black', linestyle='dashdot')
ax.set_xlim(2.9, 500)
ax.set_ylim(0, np.pi)
ax.set_yticks([0, 0.5*np.pi, np.pi])
ax.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
plt.tight_layout()
plt.savefig("plots/logB_"+savename+".pdf", dpi=300, bbox_inches='tight')
plt.show()
plt.close()
