import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_evolution(folder_name):
    filename = f"{folder_name}/genalgo_log.csv"

    df = pd.read_csv(filename)

    evolution_dict = dict()

    steps = np.unique(df["Step"])

    for step in steps:
        evolution_dict[step] = df[df["Step"] == step]["Fitness"]

    fig, ax1 = plt.subplots(nrows=1, sharex="all")

    ax1.boxplot(evolution_dict.values())
    ax1.set_xticklabels(evolution_dict.keys())

    ax1.set_title(f"Fitness over generations ({folder_name})")

    plt.show()
