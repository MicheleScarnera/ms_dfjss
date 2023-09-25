import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import dfjss_misc as misc

plt.rcParams['figure.dpi'] = 300

df = pd.read_csv("wait test/fitness_log.csv")

names = ["P", "P-15", "P-30"]
what_to_plot = [0, 1]

cols = [c for c in df.columns if misc.begins_with(containing_string=c, contained_string="Fitness_")]

values = [np.array(df.loc[i, cols], dtype=np.float64) for i in range(len(names))]

fig, ax1 = plt.subplots(nrows=1, sharex="all")

no_bins = 50

textstr = "" # f"Mean, StdDev, (Fisher) Kurtosis"
for j, i in enumerate(what_to_plot):
    ax1.hist(values[i], bins=no_bins, density=True, alpha=0.5, label=names[i])

    if textstr != "":
        textstr += "\n"

    textstr += f"{names[i]}: "

    # textstr += f"{np.mean(values[i]):.2f}, {np.std(values[i]):.2f}, {stats.kurtosis(a=values[i], fisher=True, bias=False):.2f}"

    if j > 0:
        textstr += f"Better than baseline {np.mean(values[i] < values[what_to_plot[0]]):.0%} of the time"
    else:
        textstr += "Baseline"

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='silver', alpha=0.5)

# place a text box in upper left in axes coords
ax1.text(0.95, 0.5, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment="right", bbox=props)

plt.legend()

plt.title("JIT Penalty of non-waiting vs waiting priority function P")

plt.show()
