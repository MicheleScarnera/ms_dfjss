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


def plot_autoencoder_training(folder_name,
                              epoch_step_size=20,
                              epoch_range_fontsize="x-small",
                              zeroone_yticks_amount=None,
                              train_color="blue",
                              val_color="orange",
                              lr_color="darkred",
                              weight_color="grey",
                              subplots_sharex=False,
                              dpi=200,
                              show_plots=False, suptitles=True):

    filename = f"{folder_name}/log.csv"
    plot_title = folder_name.split("\\")[-1]

    df = pd.read_csv(filename)

    if zeroone_yticks_amount is None:
        zeroone_yticks_amount = int(-np.log2(1. - np.amax(df["Val_Accuracy"]))) + 1
        print(f"Setting zeroone_yticks_amount to {zeroone_yticks_amount}")

    fig_size_h = 9 + 0.1 * len(df)
    fig_size_v = fig_size_h / np.sqrt(2)

    def epoch_label(epoch):
        return f"{epoch}\nT: {misc.large_number_format(df['Train_TotalDatapoints'][epoch - 1])}\n V: {misc.large_number_format(df['Val_TotalDatapoints'][epoch - 1])}"

    if df["Epoch"][0] == 0:
        epoch_range = range(0, df["Epoch"][len(df) - 1] + 1, epoch_step_size)
    else:
        epoch_range = [1]
        epoch_range.extend(list(range(epoch_step_size, df["Epoch"][len(df) - 1] + 1, epoch_step_size)))

    epoch_labels = [epoch_label(epoch) for epoch in epoch_range]

    def annotate_axis_with_value(axis, y_name, color="white", valueformat=".2f", distort=False):
        bbox_props = dict(boxstyle="round", fc=color, ec="0.5", alpha=0.25, edgecolor=None)
        for i, (x, y) in enumerate(zip(df["Epoch"], distortion(df[y_name]) if distort else df[y_name])):
            if i + (df["Epoch"][0]) not in epoch_range:  # i % text_box_step_size != 0:
                continue

            value = df[y_name][i] if distort else y

            axis.annotate(f"{value:^{valueformat}}", xy=(x, y), fontsize="x-small", bbox=bbox_props)

    # Losses

    fig, ((ax_total, ax_tbd), (ax_raw, ax_reduced)) = plt.subplots(nrows=2, ncols=2, sharex=subplots_sharex)
    fig.set_size_inches(fig_size_h, fig_size_v)

    # Total criterion, weights
    ax_weight = ax_total.twinx()
    ax_weight.plot(df["Epoch"], df["Criterion_Weight_Raw"], color=weight_color, alpha=0.5)

    ax_weight.set_ylim((0, 1))

    ax_total.plot(df["Epoch"], df["Train_Total_Criterion"], color=train_color, label="Train")
    ax_total.plot(df["Epoch"], df["Val_Total_Criterion"], color=val_color, label="Validation")
    ax_total.plot([0], [0], color=weight_color, label="Raw criterion weight (de facto)")

    annotate_axis_with_value(ax_total, "Train_Total_Criterion", train_color)

    ax_total.set_xticks(ticks=epoch_range, labels=epoch_labels, fontsize=epoch_range_fontsize)

    ax_total.set_title("Total Criterion")
    ax_total.set_xlabel("Epoch (Training Data, Validation Data)")
    ax_total.set_ylabel("Total Criterion")
    ax_total.legend()

    # TBD

    ax_tbd.set_title("TBD")

    # Raw Criterion
    ax_raw.plot(df["Epoch"], df["Train_Raw_Criterion"], color=train_color)
    ax_raw.plot(df["Epoch"], df["Val_Raw_Criterion"], color=val_color)

    annotate_axis_with_value(ax_raw, "Train_Raw_Criterion", train_color)

    ax_raw.set_xticks(ticks=epoch_range, labels=epoch_labels, fontsize=epoch_range_fontsize)

    ax_raw.set_title("Raw Criterion")
    ax_raw.set_xlabel("Epoch (Training Data, Validation Data)")
    ax_raw.set_ylabel("Raw Criterion")

    # Reduced Criterion
    ax_reduced.plot(df["Epoch"], df["Train_Reduced_Criterion"], color=train_color)
    ax_reduced.plot(df["Epoch"], df["Val_Reduced_Criterion"], color=val_color)

    annotate_axis_with_value(ax_reduced, "Train_Reduced_Criterion", train_color)

    ax_reduced.set_xticks(ticks=epoch_range, labels=epoch_labels, fontsize=epoch_range_fontsize)

    ax_reduced.set_title("Reduced Criterion")
    ax_reduced.set_xlabel("Epoch (Training Data, Validation Data)")
    ax_reduced.set_ylabel("Reduced Criterion")

    if suptitles:
        plt.suptitle(f"Criterion ({plot_title})")

    fig.savefig(f"{folder_name}/criterion_plot", dpi=dpi)

    if show_plots:
        plt.show()

    # [0,1] metrics

    fig, ((ax_syntax, ax_accuracy), (ax_tbd2, ax_perfects)) = plt.subplots(nrows=2, ncols=2, sharex=subplots_sharex)
    fig.set_size_inches(fig_size_h, fig_size_v)

    @np.vectorize
    def distortion(x):
        return - np.log(1. - x)

    zerone_y_labels_numeric = 1. - np.array([2 ** -i for i in range(0, zeroone_yticks_amount)])
    zerone_y_labels = ["{:.2%}".format(x) for x in zerone_y_labels_numeric]
    zerone_y_ticks = distortion(zerone_y_labels_numeric)

    # Syntax Score, Valid
    ax_valid = ax_syntax.twinx()

    valid_linestyle = "dotted"

    ax_valid.plot(df["Epoch"], distortion(df["Train_Valid"]), color=train_color, linestyle=valid_linestyle)
    ax_valid.plot(df["Epoch"], distortion(df["Val_Valid"]), color=val_color, linestyle=valid_linestyle)

    annotate_axis_with_value(ax_valid, "Train_Valid", train_color, ".2%", True)

    ax_valid.set_ylabel("Valid")
    ax_valid.set_yticks(zerone_y_ticks, zerone_y_labels)

    ax_syntax.plot(df["Epoch"], distortion(df["Train_SyntaxScore"]), color=train_color, label="Syntax Score",
                   linestyle="solid")
    ax_syntax.plot(df["Epoch"], distortion(df["Val_SyntaxScore"]), color=val_color)
    ax_syntax.plot([0], [0], color=train_color, label="Valid", linestyle=valid_linestyle)

    annotate_axis_with_value(ax_syntax, "Train_SyntaxScore", train_color, ".2%", True)

    ax_syntax.set_xticks(ticks=epoch_range, labels=epoch_labels, fontsize=epoch_range_fontsize)

    ax_syntax.set_title("Syntax Score, Valid")
    ax_syntax.set_xlabel("Epoch (Training Data, Validation Data)")
    ax_syntax.set_ylabel("Syntax Score")
    ax_syntax.set_yticks(zerone_y_ticks, zerone_y_labels)
    ax_syntax.grid(which="major")
    ax_syntax.legend()

    # Accuracy

    ax_accuracy.plot(df["Epoch"], distortion(df["Train_Accuracy"]), color=train_color)
    ax_accuracy.plot(df["Epoch"], distortion(df["Val_Accuracy"]), color=val_color)

    annotate_axis_with_value(ax_accuracy, "Train_Accuracy", train_color, ".2%", True)
    annotate_axis_with_value(ax_accuracy, "Val_Accuracy", val_color, ".2%", True)

    ax_accuracy.set_xticks(ticks=epoch_range, labels=epoch_labels, fontsize=epoch_range_fontsize)

    ax_accuracy.set_title("Accuracy")
    ax_accuracy.set_xlabel("Epoch (Training Data, Validation Data)")
    ax_accuracy.set_ylabel("Accuracy")
    ax_accuracy.set_yticks(zerone_y_ticks, zerone_y_labels)
    ax_accuracy.grid(which="major")

    # TBD

    ax_tbd2.set_title("TBD")

    # Perfects
    def annotate_lr_decreases(axis):
        bbox_props = dict(boxstyle="roundtooth", fc="w", ec="0.5", alpha=0.7, edgecolor=None)
        for i, (epoch, lr) in enumerate(zip(df["Epoch"], df["Train_LR"])):
            if i == 0:
                continue

            prev_lr = df["Train_LR"][i-1]

            if lr < prev_lr:
                axis.axvline(x=epoch - 0.5, linestyle="dotted", color=lr_color, alpha=0.25)

                axis.annotate(f"Epoch {epoch}+: {lr:.1e}", xy=(epoch, np.sqrt(lr*prev_lr)), fontsize="x-small", bbox=bbox_props)

    if "Train_LR" in df.columns:
        ax_perfects_lr = ax_perfects.twinx()
        ax_perfects_lr.semilogy(df["Epoch"], df["Train_LR"],
                                color=lr_color, alpha=.75,
                                label="Learning Rate", drawstyle="steps-mid")

        annotate_lr_decreases(ax_perfects_lr)

        ax_perfects_lr.set_ylabel("Learning Rate")
        ax_perfects_lr.legend()

    ax_perfects.plot(df["Epoch"], distortion(df["Train_Perfects"]), color=train_color)
    ax_perfects.plot(df["Epoch"], distortion(df["Val_Perfects"]), color=val_color)

    annotate_axis_with_value(ax_perfects, "Train_Perfects", train_color, ".2%", True)
    annotate_axis_with_value(ax_perfects, "Val_Perfects", val_color, ".2%", True)

    ax_perfects.set_xticks(ticks=epoch_range, labels=epoch_labels, fontsize=epoch_range_fontsize)

    ax_perfects.set_title("Perfects")
    ax_perfects.set_xlabel("Epoch (Training Data, Validation Data)")
    ax_perfects.set_ylabel("Perfects")
    ax_perfects.set_yticks(zerone_y_ticks, zerone_y_labels)
    ax_perfects.grid(which="major")

    if suptitles:
        plt.suptitle(f"[0,1] metrics ({plot_title})")

    fig.savefig(f"{folder_name}/zero_one_metrics_plot", dpi=dpi)

    if show_plots:
        plt.show()
