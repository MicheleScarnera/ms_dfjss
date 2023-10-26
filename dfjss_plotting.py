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


def plot_autoencoder_training(folder_name, epoch_step_size=4, text_box_step_size=4, dpi=400, show_plots=False):
    filename = f"{folder_name}/log.csv"
    plot_title = folder_name.split("\\")[-1]

    train_color = "blue"
    val_color = "orange"

    df = pd.read_csv(filename)

    fig_size_h = 18 + 0.3 * len(df)

    def annotate_axis_with_value(axis, y_name):
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.7, edgecolor=None)
        for i, (x, y) in enumerate(zip(df["Epoch"], df[y_name])):
            if i % text_box_step_size != 0:
                continue

            axis.annotate(f"{y:.2f}", xy=(x, y), fontsize="small", bbox=bbox_props)

    def epoch_label(epoch):
        return f"{epoch}\nT: {misc.large_number_format(df['Train_TotalDatapoints'][epoch-1])}\n V: {misc.large_number_format(df['Val_TotalDatapoints'][epoch-1])}"

    # Losses

    fig, ((ax_total, ax_weight), (ax_raw, ax_reduced)) = plt.subplots(nrows=2, ncols=2, sharex="all")
    fig.set_size_inches(fig_size_h, 15)

    # Total criterion
    ax_total.plot(df["Epoch"], df["Train_Total_Criterion"], color=train_color, label="Train")
    ax_total.plot(df["Epoch"], df["Val_Total_Criterion"], color=val_color, label="Validation")

    annotate_axis_with_value(ax_total, "Train_Total_Criterion")

    epoch_range = range(1, df["Epoch"][len(df) - 1] + 1, epoch_step_size)
    ax_total.set_xticks(ticks=epoch_range, labels=[epoch_label(epoch) for epoch in epoch_range])

    ax_total.set_title("Total Criterion")
    ax_total.legend()

    # Criterion weights

    ax_weight.plot(df["Epoch"], df["Criterion_Weight_Raw"], color="grey")

    ax_weight.set_ylim((0, 1))

    annotate_axis_with_value(ax_weight, "Criterion_Weight_Raw")

    ax_weight.set_title("Raw weights (de facto)")

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

    plt.suptitle(f"Criterion ({plot_title})")

    fig.savefig(f"{folder_name}/criterion_plot", dpi=dpi)

    if show_plots:
        plt.show()

    # [0,1] metrics

    fig, ((ax_syntax, ax_accuracy), (ax_valid, ax_perfects)) = plt.subplots(nrows=2, ncols=2, sharex="all")
    fig.set_size_inches(fig_size_h, 15)

    # Syntax Score
    ax_syntax.plot(df["Epoch"], df["Train_SyntaxScore"], color=train_color, label="Train")
    ax_syntax.plot(df["Epoch"], df["Val_SyntaxScore"], color=val_color, label="Validation")

    annotate_axis_with_value(ax_syntax, "Train_SyntaxScore")

    ax_syntax.set_xticks(ticks=epoch_range, labels=[epoch_label(epoch) for epoch in epoch_range])

    ax_syntax.set_title("Syntax Score")
    ax_syntax.legend()

    # Accuracy

    ax_accuracy.plot(df["Epoch"], df["Train_Accuracy"], color=train_color)
    ax_accuracy.plot(df["Epoch"], df["Val_Accuracy"], color=val_color)

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

    plt.suptitle(f"[0,1] metrics ({plot_title})")

    fig.savefig(f"{folder_name}/zero_one_metrics_plot", dpi=dpi)

    if show_plots:
        plt.show()
