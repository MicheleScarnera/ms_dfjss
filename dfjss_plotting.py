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


def annotate_axis_with_value(axis, df, y_name, epoch_range, color="white", valueformat=".2f", distort=False,
                             distortion=lambda x: x):
    bbox_props = dict(boxstyle="round", fc=color, ec="0.5", alpha=0.25, edgecolor=None)
    for i, (x, y) in enumerate(zip(df["Epoch"], distortion(df[y_name]) if distort else df[y_name])):
        if i + (df["Epoch"][0]) not in epoch_range:  # i % text_box_step_size != 0:
            continue

        value = df[y_name][i] if distort else y

        axis.annotate(f"{value:^{valueformat}}", xy=(x, y), fontsize="x-small", bbox=bbox_props)


def plot_autoencoder_training(folder_name,
                              epoch_ticks_amount=5,
                              epoch_range_fontsize="x-small",
                              zeroone_yticks_amount=None,
                              zeroone_yticks_amount_max=16,
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
        try:
            zeroone_yticks_amount = int(-np.log2(1. - np.amax(df["Val_Autoencoder_Accuracy"]))) + 1
        except OverflowError:
            zeroone_yticks_amount = zeroone_yticks_amount_max

        print(f"Setting zeroone_yticks_amount to {zeroone_yticks_amount}")

    fig_size_h = 8 + 0.25 * np.log2(len(df))
    fig_size_v = fig_size_h / np.sqrt(2)

    def epoch_label(epoch):
        return f"{epoch}\nT: {misc.large_number_format(df['Train_Autoencoder_TotalDatapoints'][epoch - 1])}\n V: {misc.large_number_format(df['Val_Autoencoder_TotalDatapoints'][epoch - 1])}"

    epoch_step_size = len(df) // epoch_ticks_amount

    if df["Epoch"][0] == 0:
        epoch_range = range(0, df["Epoch"][len(df) - 1] + 1, epoch_step_size)
    else:
        epoch_range = [1]
        epoch_range.extend(list(range(epoch_step_size, df["Epoch"][len(df) - 1] + 1, epoch_step_size)))

    epoch_labels = [epoch_label(epoch) for epoch in epoch_range]

    # fig, ((ax_total, ax_tbd), (ax_raw, ax_reduced)) = plt.subplots(nrows=2, ncols=2, sharex=subplots_sharex)

    # Total criterion, weights
    fig, ax_total = plt.subplots()
    fig.set_size_inches(fig_size_h, fig_size_v)

    ax_weight = ax_total.twinx()
    ax_weight.plot(df["Epoch"], df["Criterion_Weight_Raw"], color=weight_color, alpha=0.5)

    ax_weight.set_ylim((0, 1))

    ax_total.plot(df["Epoch"], df["Train_Autoencoder_Total_Criterion"], color=train_color, label="Train")
    ax_total.plot(df["Epoch"], df["Val_Autoencoder_Total_Criterion"], color=val_color, label="Validation")
    ax_total.plot([0], [0], color=weight_color, label="Raw criterion weight (de facto)")

    annotate_axis_with_value(ax_total, df, "Train_Autoencoder_Total_Criterion", epoch_range, train_color)

    ax_total.set_xticks(ticks=epoch_range, labels=epoch_labels, fontsize=epoch_range_fontsize)

    ax_total.set_title("Total Criterion")
    ax_total.set_xlabel("Epoch (Training Data, Validation Data)")
    ax_total.set_ylabel("Total Criterion")
    ax_total.legend()

    if suptitles:
        plt.suptitle(f"{plot_title}")

    fig.savefig(f"{folder_name}/plot_total", dpi=dpi)

    if show_plots:
        plt.show()

    # Raw Criterion
    fig, ax_raw = plt.subplots()
    fig.set_size_inches(fig_size_h, fig_size_v)

    ax_raw.plot(df["Epoch"], df["Train_Autoencoder_Raw_Criterion"], color=train_color)
    ax_raw.plot(df["Epoch"], df["Val_Autoencoder_Raw_Criterion"], color=val_color)

    annotate_axis_with_value(ax_raw, df, "Train_Autoencoder_Raw_Criterion", epoch_range, train_color)

    ax_raw.set_xticks(ticks=epoch_range, labels=epoch_labels, fontsize=epoch_range_fontsize)

    ax_raw.set_title("Raw Criterion")
    ax_raw.set_xlabel("Epoch (Training Data, Validation Data)")
    ax_raw.set_ylabel("Raw Criterion")

    if suptitles:
        plt.suptitle(f"{plot_title}")

    fig.savefig(f"{folder_name}/plot_raw", dpi=dpi)

    if show_plots:
        plt.show()

    # Reduced Criterion
    fig, ax_reduced = plt.subplots()
    fig.set_size_inches(fig_size_h, fig_size_v)

    ax_reduced.plot(df["Epoch"], df["Train_Autoencoder_Reduced_Criterion"], color=train_color)
    ax_reduced.plot(df["Epoch"], df["Val_Autoencoder_Reduced_Criterion"], color=val_color)

    annotate_axis_with_value(ax_reduced, df, "Train_Autoencoder_Reduced_Criterion", epoch_range, train_color)

    ax_reduced.set_xticks(ticks=epoch_range, labels=epoch_labels, fontsize=epoch_range_fontsize)

    ax_reduced.set_title("Reduced Criterion")
    ax_reduced.set_xlabel("Epoch (Training Data, Validation Data)")
    ax_reduced.set_ylabel("Reduced Criterion")

    if suptitles:
        plt.suptitle(f"{plot_title}")

    fig.savefig(f"{folder_name}/plot_reduced", dpi=dpi)

    if show_plots:
        plt.show()

    # [0,1] metrics

    # fig, ((ax_syntax, ax_accuracy), (ax_tbd2, ax_perfects)) = plt.subplots(nrows=2, ncols=2, sharex=subplots_sharex)

    distortion_max_value = zeroone_yticks_amount

    @np.vectorize
    def distortion(x):
        return - np.log2(1. - x) if x < 1. else distortion_max_value

    zerone_y_labels_numeric = 1. - np.array([2 ** -i for i in range(0, zeroone_yticks_amount)])
    zerone_y_labels = ["{:.2%}".format(x) for x in zerone_y_labels_numeric]
    zerone_y_ticks = distortion(zerone_y_labels_numeric)

    def annotate_lr_decreases(axis):
        bbox_props = dict(boxstyle="roundtooth", fc="w", ec="0.5", alpha=0.7, edgecolor=None)
        for i, (epoch, lr) in enumerate(zip(df["Epoch"], df["Train_Autoencoder_LR"])):
            if i == 0:
                continue

            prev_lr = df["Train_Autoencoder_LR"][i - 1]

            if lr < prev_lr:
                axis.axvline(x=epoch - 0.5, linestyle="dotted", color=lr_color, alpha=0.25)

                axis.annotate(f"Epoch {epoch}+: {lr:.1e}", xy=(epoch, np.sqrt(lr * prev_lr)), fontsize="x-small",
                              bbox=bbox_props)

    # Accuracy

    fig, ax_accuracy = plt.subplots()
    fig.set_size_inches(fig_size_h, fig_size_v)

    if "Train_Autoencoder_LR" in df.columns:
        ax_perfects_lr = ax_accuracy.twinx()
        ax_perfects_lr.semilogy(df["Epoch"], df["Train_Autoencoder_LR"],
                                color=lr_color, alpha=.75,
                                label="Learning Rate", drawstyle="steps-mid")

        annotate_lr_decreases(ax_perfects_lr)

        ax_perfects_lr.set_ylabel("Learning Rate")
        ax_perfects_lr.legend()

    ax_accuracy.plot(df["Epoch"], distortion(df["Train_Autoencoder_Accuracy"]), color=train_color)
    ax_accuracy.plot(df["Epoch"], distortion(df["Val_Autoencoder_Accuracy"]), color=val_color)

    annotate_axis_with_value(ax_accuracy, df, "Train_Autoencoder_Accuracy", epoch_range, train_color, ".2%", True,
                             distortion)
    annotate_axis_with_value(ax_accuracy, df, "Val_Autoencoder_Accuracy", epoch_range, val_color, ".2%", True,
                             distortion)

    ax_accuracy.set_xticks(ticks=epoch_range, labels=epoch_labels, fontsize=epoch_range_fontsize)

    ax_accuracy.set_title("Accuracy")
    ax_accuracy.set_xlabel("Epoch (Training Data, Validation Data)")
    ax_accuracy.set_ylabel("Accuracy")
    ax_accuracy.set_yticks(zerone_y_ticks, zerone_y_labels)
    ax_accuracy.grid(which="major")

    if suptitles:
        plt.suptitle(f"{plot_title}")

    fig.savefig(f"{folder_name}/plot_accuracy", dpi=dpi)

    if show_plots:
        plt.show()

    # Perfects

    fig, ax_perfects = plt.subplots()
    fig.set_size_inches(fig_size_h, fig_size_v)

    ax_perfects.plot(df["Epoch"], distortion(df["Train_Autoencoder_Perfects"]), color=train_color)
    ax_perfects.plot(df["Epoch"], distortion(df["Val_Autoencoder_Perfects"]), color=val_color)

    annotate_axis_with_value(ax_perfects, df, "Train_Autoencoder_Perfects", epoch_range, train_color, ".2%", True,
                             distortion)
    annotate_axis_with_value(ax_perfects, df, "Val_Autoencoder_Perfects", epoch_range, val_color, ".2%", True,
                             distortion)

    ax_perfects.set_xticks(ticks=epoch_range, labels=epoch_labels, fontsize=epoch_range_fontsize)

    ax_perfects.set_title("Perfects")
    ax_perfects.set_xlabel("Epoch (Training Data, Validation Data)")
    ax_perfects.set_ylabel("Perfects")
    ax_perfects.set_yticks(zerone_y_ticks, zerone_y_labels)
    ax_perfects.grid(which="major")

    if suptitles:
        plt.suptitle(f"{plot_title}")

    fig.savefig(f"{folder_name}/plot_perfects", dpi=dpi)

    if show_plots:
        plt.show()


def plot_reward_model_training(folder_name,
                               baseline_values=None,
                               epoch_ticks_amount=5,
                               epoch_range_fontsize="x-small",
                               train_color="blue",
                               val_color="orange",
                               baseline_color="purple",
                               baseline_alpha=0.5,
                               baseline_linestyle="dashed",
                               dpi=200,
                               show_plots=False,
                               suptitles=True):
    filename = f"{folder_name}/log.csv"
    plot_title = folder_name.split("\\")[-1]

    df = pd.read_csv(filename)

    fig_size_h = 8 + 0.25 * np.log2(len(df))
    fig_size_v = fig_size_h / np.sqrt(2)

    epoch_step_size = len(df) // epoch_ticks_amount

    def epoch_label(epoch):
        return f"{epoch}"

    if df["Epoch"][0] == 0:
        epoch_range = range(0, df["Epoch"][len(df) - 1] + 1, epoch_step_size)
    else:
        epoch_range = [1]
        epoch_range.extend(list(range(epoch_step_size, df["Epoch"][len(df) - 1] + 1, epoch_step_size)))

    epoch_labels = [epoch_label(epoch) for epoch in epoch_range]

    loss_names = ("L1", "SmoothL1", "L2")

    for i in range(3):
        loss_name = loss_names[i]
        fig, ax_loss = plt.subplots()
        fig.set_size_inches(fig_size_h, fig_size_v)

        train_col = f"Train_{loss_name}_Loss"
        val_col = f"Val_{loss_name}_Loss"

        ax_loss.plot(df["Epoch"], df[train_col], color=train_color, label="Train")
        ax_loss.plot(df["Epoch"], df[val_col], color=val_color, label="Validation")

        if baseline_values is not None and baseline_values[i] is not None:
            ax_loss.axhline(baseline_values[i], color=baseline_color, alpha=baseline_alpha, linestyle=baseline_linestyle, label="Baseline")

        annotate_axis_with_value(ax_loss, df, train_col, epoch_range, train_color)
        annotate_axis_with_value(ax_loss, df, val_col, epoch_range, val_color)

        ax_loss.set_xticks(ticks=epoch_range, labels=epoch_labels, fontsize=epoch_range_fontsize)

        ax_loss.set_title(loss_name)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel(loss_name)
        ax_loss.grid(which="major")

        ax_loss.legend()

        if suptitles:
            plt.suptitle(f"{plot_title}")

        fig.savefig(f"{folder_name}/plot_{loss_name}", dpi=dpi)

        if show_plots:
            plt.show()
