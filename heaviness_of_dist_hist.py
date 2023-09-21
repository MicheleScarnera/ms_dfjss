import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import dfjss_misc as misc

plt.rcParams['figure.dpi'] = 300

df_sixty = pd.read_csv("FIFO cooldown 0 60/fitness_log.csv")
df_fifteen = pd.read_csv("FIFO cooldown 0 15/fitness_log.csv")

cols = [c for c in df_sixty.columns if misc.begins_with(containing_string=c, contained_string="Fitness_")]

values_sixty = np.array(df_sixty.loc[0, cols], dtype=np.float64)
values_fifteen = np.array(df_fifteen.loc[0, cols], dtype=np.float64)

fig, ax1 = plt.subplots(nrows=1, sharex="all")

no_bins = 50

ax1.hist(values_sixty, bins=no_bins, density=True, alpha=0.5, label="U(0,60)")
ax1.hist(values_fifteen, bins=no_bins, density=True, alpha=0.5, label="U(0,15)")

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='silver', alpha=0.5)

textstr = f"""Mean, StdDev, (Fisher) Kurtosis
U(0,60): {np.mean(values_sixty):.2f}, {np.std(values_sixty):.2f}, {stats.kurtosis(a=values_sixty, fisher=True, bias=False):.2f}
U(0,15): {np.mean(values_fifteen):.2f}, {np.std(values_fifteen):.2f}, {stats.kurtosis(a=values_fifteen, fisher=True, bias=False):.2f}"""

# place a text box in upper left in axes coords
ax1.text(0.95, 0.5, textstr, transform=ax1.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment="right", bbox=props)

plt.legend()

plt.title("FIFO individual's JIT penalty with different cooldown/windup distributions")

plt.show()
