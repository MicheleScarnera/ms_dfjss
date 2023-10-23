import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import dfjss_misc as misc


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


def plot_autoencoder_training(folder_name, dpi=400):
    filename = f"{folder_name}/log.csv"

    train_color = "blue"
    val_color = "orange"

    df = pd.read_csv(filename)

    def annotate_axis_with_value(axis, y_name):
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.7, edgecolor=None)
        for x, y in zip(df["Epoch"], df[y_name]):
            axis.annotate(f"{y:.2f}", xy=(x, y), fontsize="small", bbox=bbox_props)

    # Losses

    fig, ((ax_total, ax_weight), (ax_raw, ax_reduced)) = plt.subplots(nrows=2, ncols=2, sharex="all")
    fig.set_size_inches(20, 15)

    # Total criterion
    ax_total.plot(df["Epoch"], df["Train_Total_Criterion"], color=train_color, label="Train")
    ax_total.plot(df["Epoch"], df["Val_Total_Criterion"], color=val_color, label="Validation")

    annotate_axis_with_value(ax_total, "Train_Total_Criterion")

    epoch_range = range(1, df["Epoch"][len(df) - 1] + 1)
    ax_total.set_xticks(ticks=epoch_range, labels=[f"{epoch}\n{misc.large_number_format(data_amount)}" for epoch, data_amount in zip(df["Epoch"], df["Train_TotalDatapoints"])])

    ax_total.set_title("Total Criterion")
    ax_total.legend()

    # Criterion weights

    ax_weight.plot(df["Epoch"], df["Criterion_Weight_Raw"], color="grey")

    ax_weight.set_ylim((0, 1))

    annotate_axis_with_value(ax_weight, "Criterion_Weight_Raw")

    ax_weight.set_title("Raw weights")

    # Raw Criterion
    ax_raw.plot(df["Epoch"], df["Train_Raw_Criterion"], color=train_color)
    ax_raw.plot(df["Epoch"], df["Val_Raw_Criterion"], color=val_color)

    annotate_axis_with_value(ax_raw, "Train_Raw_Criterion")

    ax_raw.set_title("Raw criterion")

    # Reduced Criterion
    ax_reduced.plot(df["Epoch"], df["Train_Reduced_Criterion"], color=train_color)
    ax_reduced.plot(df["Epoch"], df["Val_Reduced_Criterion"], color=val_color)

    annotate_axis_with_value(ax_reduced, "Train_Reduced_Criterion")

    ax_reduced.set_title("Reduced criterion")

    plt.suptitle(f"Criterion ({folder_name})")

    fig.savefig(f"{folder_name}/criterion_plot", dpi=dpi)

    plt.show()

    # [0,1] metrics

    fig, ((ax_syntax, ax_accuracy), (ax_valid, ax_perfects)) = plt.subplots(nrows=2, ncols=2, sharex="all")
    fig.set_size_inches(20, 15)

    # Syntax Score
    ax_syntax.plot(df["Epoch"], df["Train_SyntaxScore"], color=train_color, label="Train")
    ax_syntax.plot(df["Epoch"], df["Val_SyntaxScore"], color=val_color, label="Validation")

    annotate_axis_with_value(ax_syntax, "Train_SyntaxScore")

    epoch_range = range(1, df["Epoch"][len(df) - 1] + 1)
    ax_syntax.set_xticks(ticks=epoch_range,
                        labels=[f"{epoch}\n{misc.large_number_format(data_amount)}" for epoch, data_amount in
                                zip(df["Epoch"], df["Train_TotalDatapoints"])])

    ax_syntax.set_title("Syntax Score")
    ax_syntax.legend()

    # Accuracy

    ax_accuracy.plot(df["Epoch"], df["Train_Accuracy"], color=train_color)
    ax_accuracy.plot(df["Epoch"], df["Val_Accuracy"], color=val_color)

    ax_accuracy.set_ylim((0, 1))

    annotate_axis_with_value(ax_accuracy, "Train_Accuracy")

    ax_accuracy.set_title("Accuracy")

    # Valid
    ax_valid.plot(df["Epoch"], df["Train_Valid"], color=train_color)
    ax_valid.plot(df["Epoch"], df["Val_Valid"], color=val_color)

    annotate_axis_with_value(ax_valid, "Train_Valid")

    ax_valid.set_title("Valid")

    # Perfects
    ax_perfects.plot(df["Epoch"], df["Train_Perfects"], color=train_color)
    ax_perfects.plot(df["Epoch"], df["Val_Perfects"], color=val_color)

    annotate_axis_with_value(ax_perfects, "Train_Perfects")

    ax_perfects.set_title("Perfects")

    plt.suptitle(f"[0,1] metrics ({folder_name})")

    fig.savefig(f"{folder_name}/zero_one_metrics_plot", dpi=dpi)

    plt.show()