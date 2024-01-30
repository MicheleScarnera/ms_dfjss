import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import kurtosis

import dfjss_misc as misc

aspect_ratio = 1.6
dpi = 200

plt.rcParams['figure.dpi'] = dpi


def plot_wait_dist(folder_name,
                   show_plot=False):
    df = pd.read_csv(f"{folder_name}/fitness_log.csv")

    names = ["P", "P-2", "P-4", "P-8", "P-16", "P-32", "P-64"]
    what_to_plot = [[0, 1], [0, 5]]
    titles = ["JIT Penalty of non-waiting vs waiting priority function P",
              "JIT Penalty of non-waiting vs overly patient priority function P"]
    file_names = ["nonwaiting_vs_waiting", "nonwaiting_vs_overwaiting"]

    cols = [c for c in df.columns if misc.begins_with(string=c, prefix="Fitness_")]

    values = [np.array(df.loc[i, cols], dtype=np.float64) for i in range(len(names))]

    fig_size_h = 8
    fig_size_v = fig_size_h / aspect_ratio

    for w, wtp in enumerate(what_to_plot):
        fig, ax1 = plt.subplots(nrows=1, sharex="all")

        fig.set_size_inches(fig_size_h, fig_size_v)
        plt.xscale("log")

        x0 = 10 ** 2
        x1 = 10 ** 3.5

        bins = np.floor(np.geomspace(x0, x1, num=100))

        textstr = ""
        for j, i in enumerate(wtp):
            ax1.set_xlim(x0, x1)
            ax1.hist(values[i], bins=bins, density=True, alpha=0.5, label=names[i])

            if textstr != "":
                textstr += "\n"

            textstr += f"{names[i]}: "

            # textstr += f"{np.mean(values[i]):.2f}, {np.std(values[i]):.2f}, {stats.kurtosis(a=values[i], fisher=True, bias=False):.2f}"

            if j > 0:
                textstr += f"Better than baseline {np.mean(values[i] < values[wtp[0]]):.0%} of the time"
            else:
                textstr += "Baseline"

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='silver', alpha=0.5)

        # place a text box in upper left in axes coords
        ax1.text(0.95, 0.5, textstr, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='bottom', horizontalalignment="right", bbox=props)

        plt.legend()

        plt.title(titles[w])

        fig.savefig(f"{folder_name}/{file_names[w]}", dpi=dpi)

        if show_plot:
            plt.show()


def plot_heaviness_dist(folder_name_sixty,
                        folder_name_fifteen,
                        show_plot=False):
    df_sixty = pd.read_csv(f"{folder_name_sixty}/fitness_log.csv")
    df_fifteen = pd.read_csv(f"{folder_name_fifteen}/fitness_log.csv")

    cols = [c for c in df_sixty.columns if misc.begins_with(string=c, prefix="Fitness_")]

    values_sixty = np.array(df_sixty.loc[0, cols], dtype=np.float64)
    values_fifteen = np.array(df_fifteen.loc[0, cols], dtype=np.float64)

    fig, ax1 = plt.subplots(nrows=1, sharex="all")
    plt.xscale("log")

    x0 = 10 ** 2
    x1 = 10 ** 3.5

    bins = np.floor(np.geomspace(x0, x1, num=100))

    ax1.hist(values_sixty, bins=bins, density=True, alpha=0.5, label="U(0,60)")
    ax1.hist(values_fifteen, bins=bins, density=True, alpha=0.5, label="U(0,15)")

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='silver', alpha=0.5)

    textstr = f"""Mean, StdDev, (Fisher) Kurtosis
    U(0,60): {np.mean(values_sixty):.2f}, {np.std(values_sixty):.2f}, {kurtosis(a=values_sixty, fisher=True, bias=False):.2f}
    U(0,15): {np.mean(values_fifteen):.2f}, {np.std(values_fifteen):.2f}, {kurtosis(a=values_fifteen, fisher=True, bias=False):.2f}"""

    # place a text box in upper left in axes coords
    ax1.text(0.95, 0.5, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment="right", bbox=props)

    plt.legend()

    plt.title("FIFO individual's JIT penalty with different cooldown/windup distributions")

    fig.savefig(f"cooldown_windup_heavy_tail", dpi=dpi)

    if show_plot:
        plt.show()


def plot_evolution(folder_name,
                   show_plot=False,
                   best_color="dodgerblue", best_lw=2,
                   median_color="grey", median_alpha=0.75, median_ls="dashed",
                   mean_color="brown", mean_alpha=0.5, mean_ls="dotted",
                   others_color="teal", others_alpha=0.15):
    filename = f"{folder_name}/genalgo_log.csv"

    df = pd.read_csv(filename)

    fig_size_h = 8 + 0.25 * np.log2(len(df))
    fig_size_v = fig_size_h / aspect_ratio

    evolution_dict = dict()

    steps = np.unique(df["Step"])

    x = np.arange(len(steps))
    best_each_step = np.empty(shape=len(steps))
    mean_each_step = np.empty(shape=len(steps))
    median_each_step = np.empty(shape=len(steps))
    third_quartile_each_step = np.empty(shape=len(steps))

    for step in steps:
        evolution_dict[step] = df[df["Step"] == step]["Fitness"]
        best_each_step[step - 1] = np.amin(evolution_dict[step])
        mean_each_step[step - 1] = np.mean(evolution_dict[step])
        median_each_step[step - 1] = np.median(evolution_dict[step])
        third_quartile_each_step[step - 1] = np.quantile(evolution_dict[step], q=0.75)

    fig, ax1 = plt.subplots(nrows=1, sharex="all")
    fig.set_size_inches(fig_size_h, fig_size_v)

    # ax1.boxplot(evolution_dict.values())
    ax1.fill_between(x, y1=best_each_step, y2=third_quartile_each_step,
                     color=others_color, alpha=others_alpha, label="[Min, Q3]")

    ax1.plot(x, mean_each_step,
             color=mean_color, alpha=mean_alpha, ls=mean_ls, label="Mean")
    ax1.plot(x, median_each_step,
             color=median_color, alpha=median_alpha, ls=median_ls, label="Median")
    ax1.plot(x, best_each_step,
             color=best_color, lw=best_lw, label="Minimum")

    ax1.set_xticks(x, x + 1)

    ax1.grid(axis="x", alpha=0.5)

    ax1.legend()
    ax1.set_title(f"Fitness over generations ({folder_name})")

    fig.savefig(f"{folder_name}/plot_evolution", dpi=dpi)

    if show_plot:
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
    fig_size_v = fig_size_h / aspect_ratio

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
                               show_plots=False,
                               suptitles=True):
    filename = f"{folder_name}/log.csv"
    plot_title = folder_name.split("\\")[-1]

    df = pd.read_csv(filename)

    fig_size_h = 8 + 0.25 * np.log2(len(df))
    fig_size_v = fig_size_h / aspect_ratio

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
            ax_loss.axhline(baseline_values[i], color=baseline_color, alpha=baseline_alpha,
                            linestyle=baseline_linestyle, label="Baseline")

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
