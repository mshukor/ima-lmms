import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import argparse
import os
from functools import partial
from utils import read_hidden_states_from_folders, read_hidden_states_from_folders_llava


def compute_sims_from_hs(
    all_embed_img,
    all_embed_txt,
    new_labels,
    all_embed_img_before,
    all_embed_img_connector,
    img_layer=0,
    text_layer=0,
    connector_layer=0,
):
    img_txt_sims = []
    print("all_embed_img")
    for i in range(len(all_embed_img)):

        img = [im[img_layer] for im in all_embed_img[i]]
        txt = [im[text_layer] for im in all_embed_txt[i]]

        labl = f"{new_labels[i]} vs Text"
        img_txt_sims.append(compute_sim(img, txt) + (labl,))

    img = [im[img_layer] for im in all_embed_txt[0]]
    txt = [im[text_layer] for im in all_embed_txt[0]]
    labl = f"Text vs Text"
    img_txt_sims.append(compute_sim(img, txt) + (labl,))

    img_img_sims = []
    print("img_img_sims")
    for i in range(len(all_embed_img)):
        for j in range(len(all_embed_img)):
            img = [im[img_layer] for im in all_embed_img[i]]
            txt = [im[img_layer] for im in all_embed_img[j]]

            labl = f"{new_labels[i]} vs {new_labels[j]}"
            img_img_sims.append(compute_sim(img, txt) + (labl,))

    im_global_txt_sims = []
    all_txt_embeds = []

    for text_embed in all_embed_txt:
        all_txt_embeds += [im[text_layer] for im in text_embed]

    print("im_global_txt_sims")
    for i in range(len(all_embed_img)):

        img = [im[img_layer] for im in all_embed_img[i]]
        txt = all_txt_embeds

        labl = f"{new_labels[i]} vs All Text"
        im_global_txt_sims.append(compute_sim(img, txt) + (labl,))

    img_img_before_sims = []

    return im_global_txt_sims, img_img_sims, img_txt_sims, img_img_before_sims


def plot_sims(
    data,
    markersize=12,
    fontsize=30,
    font_offset=15,
    data_end_offset=None,
    save_path=None,
    facecolor="lavender",
    plot_name="",
):
    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams["lines.linewidth"] = 4.0
    plt.rcParams["font.family"] = "serif"  # Times New Roman serif

    plt.figure(figsize=(20, 15))
    ax = plt.axes()
    ax.set_facecolor(facecolor)

    global_sims = []
    for i, d in enumerate(data):
        sims = d[0]
        global_sim = d[1]
        label = d[2]
        global_sims.append(global_sim)
        sims = [s for s in sims if s > 0.01]
        plt.plot(sims, label=label, marker="o", linestyle="--", markersize=markersize)
        print(f"Global similarity: {global_sim}, Label: {label}")

    global_sim = sum(global_sims) / len(global_sims)
    print(f"Global similarity: {global_sim}, Label: Average")

    plt.ylabel("Cosine Similarity", fontsize=fontsize + font_offset)
    plt.xlabel("Examples", fontsize=fontsize + font_offset)
    plt.legend(fontsize=fontsize + font_offset)

    plt.grid(color="white", linewidth=4)

    plt.xticks(fontsize=fontsize + font_offset)
    plt.yticks(fontsize=fontsize + font_offset)

    if save_path is not None:
        save_plot_path = os.path.join(save_path, f"{plot_name}.jpg")
        plt.savefig(save_plot_path, bbox_inches="tight")
        print(f"saved at {save_plot_path}")

    plt.show()


def filter_im_im_sim(data):

    new_data = []
    labels = []
    for i, d in enumerate(data):
        sims = d[0]
        global_sim = d[1]
        label = d[2]
        l1, l2 = label.split(" vs ")
        inv_label = f"{l2} vs {l1}"
        if label not in labels and inv_label not in labels and l1 != l2:
            labels.append(label)
            new_data.append(d)

    return new_data


def compute_sim(list_1, list_2, unnormalized_sim=False, mode="normal"):

    all_sims = []

    for l1 in list_1:
        sims = []
        for l2 in list_2:
            l1, l2 = l1.astype(float), l2.astype(float)
            # try:

            if any([k in mode for k in ["max", "mean", "min", "median"]]):
                sim = l1 @ l2.transpose()
                norm_tensor1 = np.linalg.norm(l1, axis=1, keepdims=True)
                norm_tensor2 = np.linalg.norm(l2, axis=1, keepdims=True)
                sim = sim / (norm_tensor1 * norm_tensor2.T)

                if mode == "max":
                    sim = np.max(sim)

                elif mode == "min":
                    # try:
                    sim = np.min(sim)
                elif mode == "max_mean":
                    sim = np.max(sim, axis=-1).mean()
                elif mode == "mean":
                    sim = np.mean(sim)
                elif mode == "median":
                    sim = np.median(sim)
                elif mode == "diag_median":  # for consecutive
                    sim = np.median(np.diag(sim))
                elif mode == "diag_mean":  # for consecutive
                    sim = np.mean(np.diag(sim))
                elif mode == "diag_max":  # for consecutive
                    sim = np.max(np.diag(sim))
            else:

                sim = np.dot(l1, l2)
                sim = sim / (np.linalg.norm(l1) * np.linalg.norm(l2) + 1e-7)

            sims.append(sim)

        all_sims.append(sims)

    if list_1[0].ndim > 1:
        list_1 = [l.mean(0) for l in list_1]
        list_2 = [l.mean(0) for l in list_2]

    l1s, l2s = np.stack(list_1, axis=0).mean(0), np.stack(list_2, axis=0).mean(0)
    try:
        global_sim = np.dot(l1s, l2s) / (
            np.linalg.norm(l1s) * np.linalg.norm(l2s) + 1e-7
        )
    except ValueError:
        global_sim = 0

    return all_sims, global_sim


def plt_avg_std_consecutive_layers_sims(
    avg,
    std,
    xaxis,
    fontsize=30,
    font_offset=15,
    markersize=16,
    save_path=None,
    facecolor="lavender",
    plot_name="",
    across_epochs=False,
    epochs=[],
    y_label="Cosine Similarity",
    labels=[""],
    figsize=(20, 15),
    legend_pos="lower right",
    skip_pp=False,
):

    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams["lines.linewidth"] = 4.0
    plt.rcParams["font.family"] = "serif"  # Times New Roman serif
    plt.figure(figsize=figsize)
    ax = plt.axes()
    ax.set_facecolor(facecolor)

    if not isinstance(avg, list):
        avg = [avg]
        std = [std]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_idx = 0
    for j, (avgs, stds, lab) in enumerate(zip(avg, std, labels)):

        for i, key in enumerate(avgs.keys()):

            avg_ = avgs[key]
            std_ = stds[key]

            n = key
            # Plot the data
            averages = np.array([a[0] for a in avg_])
            std_devs = np.array([a[0] for a in std_])

            xaxis = list(range(len(averages)))

            if not skip_pp:
                color = colors[color_idx]
                color_idx += 1
                if color_idx >= len(colors):
                    color_idx = 0
                plt.plot(
                    xaxis,
                    averages,
                    marker="o",
                    linestyle="--",
                    markersize=markersize,
                    label=f"{lab}-P (n {n})",
                    color=color,
                )
                plt.fill_between(
                    xaxis,
                    averages - std_devs,
                    averages + std_devs,
                    alpha=0.2,
                    color=color,
                )
                plt.axhline(y=np.mean(averages), linestyle="--", color=color, zorder=3)

            # Plot the data

            averages = np.array([a[1] for a in avg_])
            std_devs = np.array([a[1] for a in std_])
            color = colors[color_idx]
            color_idx += 1
            if color_idx >= len(colors):
                color_idx = 0
            # plt.errorbar(xaxis, averages, yerr=std_devs, fmt='o-', capsize=5, label='Prompt vs Text')
            plt.plot(
                xaxis,
                averages,
                marker="^",
                linestyle="--",
                markersize=markersize,
                label=f"{lab}-T (n {n})",
                color=color,
            )
            plt.fill_between(
                xaxis, averages - std_devs, averages + std_devs, alpha=0.15, color=color
            )
            plt.axhline(y=np.mean(averages), linestyle="--", color=color, zorder=3)

    plt.ylabel(y_label, fontsize=fontsize + font_offset)
    plt.xlabel("Layers", fontsize=fontsize + font_offset)

    legend = plt.legend(fontsize=fontsize + font_offset, loc=legend_pos)
    legend.get_frame().set_alpha(0.5)  # Adjust alpha value as needed

    plt.grid(color="white", linewidth=4)

    plt.xticks(fontsize=fontsize + font_offset)
    plt.yticks(fontsize=fontsize + font_offset)

    if save_path is not None:
        save_plot_path = os.path.join(save_path, f"{plot_name}.jpg")
        plt.savefig(save_plot_path, bbox_inches="tight")
        print(f"saved at {save_plot_path}")

    plt.show()


def plt_avg_std_consecutive_layers_sims_across(
    avg_,
    std_,
    xaxis,
    fontsize=30,
    font_offset=15,
    markersize=16,
    save_path=None,
    facecolor="lavender",
    plot_name="",
    across_epochs=False,
    epochs=[],
    y_label="Cosine Similarity",
    labels=[""],
    figsize=(20, 15),
    legend_pos="lower right",
    skip_pp=False,
    metalabels=[""],
):

    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams["lines.linewidth"] = 4.0
    plt.rcParams["font.family"] = "serif"  # Times New Roman serif
    plt.figure(figsize=figsize)
    ax = plt.axes()
    ax.set_facecolor(facecolor)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_idx = 0

    for avg, std, labels_ in zip(avg_, std_, metalabels):

        if not isinstance(avg, list):
            avg = [avg]
            std = [std]
        for j, (avgs, stds, lab) in enumerate(zip(avg, std, labels)):

            for i, key in enumerate(avgs.keys()):

                avg_ = avgs[key]
                std_ = stds[key]

                n = key
                # Plot the data
                averages = np.array([a[0] for a in avg_])
                std_devs = np.array([a[0] for a in std_])

                xaxis = list(range(len(averages)))

                if not skip_pp:
                    color = colors[color_idx]
                    color_idx += 1
                    if color_idx >= len(colors):
                        color_idx = 0
                    plt.plot(
                        xaxis,
                        averages,
                        marker="o",
                        linestyle="--",
                        markersize=markersize,
                        label=f"{lab}-P (n {n}) {labels_}",
                        color=color,
                    )
                    plt.fill_between(
                        xaxis,
                        averages - std_devs,
                        averages + std_devs,
                        alpha=0.2,
                        color=color,
                    )
                    plt.axhline(
                        y=np.mean(averages), linestyle="--", color=color, zorder=3
                    )

                averages = np.array([a[1] for a in avg_])
                std_devs = np.array([a[1] for a in std_])
                color = colors[color_idx]
                color_idx += 1
                if color_idx >= len(colors):
                    color_idx = 0
                plt.plot(
                    xaxis,
                    averages,
                    marker="^",
                    linestyle="--",
                    markersize=markersize,
                    label=f"{lab}-T (n {n}) {labels_}",
                    color=color,
                )
                plt.fill_between(
                    xaxis,
                    averages - std_devs,
                    averages + std_devs,
                    alpha=0.15,
                    color=color,
                )
                plt.axhline(y=np.mean(averages), linestyle="--", color=color, zorder=3)

    plt.ylabel(y_label, fontsize=fontsize + font_offset)
    plt.xlabel("Layers", fontsize=fontsize + font_offset)

    legend = plt.legend(fontsize=fontsize + font_offset, loc=legend_pos)
    legend.get_frame().set_alpha(0.5)  # Adjust alpha value as needed

    plt.grid(color="white", linewidth=4)

    plt.xticks(fontsize=fontsize + font_offset)
    plt.yticks(fontsize=fontsize + font_offset)

    if save_path is not None:
        save_plot_path = os.path.join(save_path, f"{plot_name}.jpg")
        plt.savefig(save_plot_path, bbox_inches="tight")
        print(f"saved at {save_plot_path}")

    plt.show()


def compute_sims_from_hs_across_consecutive_layers(
    all_embed_img,
    all_embed_txt,
    layers=[0, 32],
    intervals=[1, 2, 4, 6, 8],
    per_modality=False,
    sim_mode="normal",
):

    compute_sim_func = partial(compute_sim, mode=sim_mode)

    avgs = {}
    stds = {}
    for interval in intervals:
        all_avg = []
        all_std = []
        for layer in tqdm(layers):
            if interval + layer > 31:
                break

            img_txt_sims = []
            img_img_sims = []
            for i in range(len(all_embed_img)):

                img = [im[layer] for im in all_embed_img[i]]
                txt = [im[layer + interval] for im in all_embed_img[i]]
                img_img_sims.append(compute_sim_func(img, txt))

                img = [im[layer] for im in all_embed_txt[i]]
                txt = [im[layer + interval] for im in all_embed_txt[i]]
                img_txt_sims.append(compute_sim_func(img, txt))

            if per_modality:
                all_img_img = []
                all_img_txt = []
                for i in range(len(all_embed_img)):
                    all_img_img.append(np.array(img_img_sims[i][0]))
                    all_img_txt.append(np.array(img_txt_sims[i][0]))
                all_avg.append(
                    [[s.mean() for s in all_img_img], [s.mean() for s in all_img_txt]]
                )
                all_std.append(
                    [[s.std() for s in all_img_img], [s.std() for s in all_img_txt]]
                )

            else:
                all_img_img = np.array([np.array(s[0]).mean() for s in img_img_sims])
                all_img_txt = np.array([np.array(s[0]).mean() for s in img_txt_sims])

                all_avg.append([np.mean(all_img_img), np.mean(all_img_txt)])
                all_std.append([np.std(all_img_img), np.std(all_img_txt)])

        avgs[interval] = all_avg
        stds[interval] = all_std

    return avgs, stds


def compute_norm(list_1, mode="avg"):

    norms = []
    for l1 in list_1:
        l1 = l1.astype(float)

        norm = np.linalg.norm(l1, axis=-1)
        if "median" in mode:
            norm = np.median(norm, axis=0)
        elif "max" in mode:
            norm = np.max(norm, axis=0)
        elif "min" in mode:
            norm = np.min(norm, axis=0)

        else:
            norm = np.linalg.norm(l1)

        norms.append(norm)

    return norms, np.stack(norms, axis=0).mean(0)


def compute_norm_from_hs_across_consecutive_layers(
    all_embed_img,
    all_embed_txt,
    layers=[0, 32],
    intervals=[1, 2, 4, 6, 8],
    per_modality=False,
    sim_mode="avg",
):

    avgs = {}
    stds = {}
    all_avg = []
    all_std = []

    compute_norm_func = partial(compute_norm, mode=sim_mode)

    for layer in tqdm(layers):

        img_txt_sims = []
        img_img_sims = []
        for i in range(len(all_embed_img)):

            img = [im[layer] for im in all_embed_img[i]]

            img_img_sims.append(compute_norm_func(img))

            img = [im[layer] for im in all_embed_txt[i]]
            img_txt_sims.append(compute_norm_func(img))

        if per_modality:
            all_img_img = []
            all_img_txt = []
            for i in range(len(all_embed_img)):
                all_img_img.append(np.array(img_img_sims[i][0]))
                all_img_txt.append(np.array(img_txt_sims[i][0]))
            all_avg.append(
                [[s.mean() for s in all_img_img], [s.mean() for s in all_img_txt]]
            )
            all_std.append(
                [[s.std() for s in all_img_img], [s.std() for s in all_img_txt]]
            )

        else:

            all_img_img = np.array([np.array(s[0]).mean() for s in img_img_sims])
            all_img_txt = np.array([np.array(s[0]).mean() for s in img_txt_sims])

            all_avg.append([np.mean(all_img_img), np.mean(all_img_txt)])
            all_std.append([np.std(all_img_img), np.std(all_img_txt)])

    avgs[1] = all_avg
    stds[1] = all_std

    return avgs, stds


def organise_per_modal_dict(all_avg, all_std):
    all_avg_ = []
    all_std_ = []
    all_keys = list(all_avg.keys())
    for j in range(len(all_avg[all_keys[0]][0][0])):
        avg_dict = {}
        std_dict = {}
        for key in all_keys:
            avg_tmp = []
            std_tmp = []
            for l in range(len(all_avg[key])):
                avg_tmp.append([all_avg[key][l][0][j], all_avg[key][l][1][j]])
                std_tmp.append([all_std[key][l][0][j], all_std[key][l][1][j]])

            avg_dict[key] = avg_tmp
            std_dict[key] = std_tmp

        all_avg_.append(avg_dict)
        all_std_.append(std_dict)
    return all_avg_, all_std_


def plt_avg_std_consecutive_layers_sims_llava(
    avg,
    std,
    xaxis,
    fontsize=30,
    font_offset=15,
    markersize=16,
    save_path=None,
    facecolor="lavender",
    plot_name="",
    across_epochs=False,
    epochs=[],
    y_label="Cosine Similarity",
    labels=[""],
    figsize=(20, 15),
    legend_pos="lower right",
    labels_to_remove=[],
    keys_to_remove=[""],
    skip_pp=False,
):

    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams["lines.linewidth"] = 4.0
    plt.rcParams["font.family"] = "serif"  # Times New Roman serif
    plt.figure(figsize=figsize)
    ax = plt.axes()
    ax.set_facecolor(facecolor)

    if not isinstance(avg, list):
        avg = [avg]
        std = [std]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    color_idx = 0
    for j, (avgs, stds, lab) in enumerate(zip(avg, std, labels)):
        if any([k in lab for k in labels_to_remove]):
            continue

        for i, key in enumerate(avgs.keys()):

            if key in keys_to_remove:
                continue

            avg_ = avgs[key]
            std_ = stds[key]

            n = key
            # Plot the data
            averages = np.array([a[0] for a in avg_])
            std_devs = np.array([a[0] for a in std_])

            xaxis = list(range(len(averages)))

            if not skip_pp:
                color = colors[color_idx]
                color_idx += 1
                if color_idx >= len(colors):
                    color_idx = 0

                plt.plot(
                    xaxis,
                    averages,
                    marker="o",
                    linestyle="--",
                    markersize=markersize,
                    label=f"{lab}-P (n {n})",
                    color=color,
                )
                plt.fill_between(
                    xaxis,
                    averages - std_devs,
                    averages + std_devs,
                    alpha=0.2,
                    color=color,
                )
                plt.axhline(y=np.mean(averages), linestyle="--", color=color, zorder=3)

            # Plot the data
            averages = np.array([a[1] for a in avg_])
            std_devs = np.array([a[1] for a in std_])

            color = colors[color_idx]
            color_idx += 1
            if color_idx >= len(colors):
                color_idx = 0
            plt.plot(
                xaxis,
                averages,
                marker="^",
                linestyle="--",
                markersize=markersize,
                label=f"{lab}-T (n {n})",
                color=color,
            )
            plt.fill_between(
                xaxis, averages - std_devs, averages + std_devs, alpha=0.15, color=color
            )
            plt.axhline(y=np.mean(averages), linestyle="--", color=color, zorder=3)

    plt.ylabel(y_label, fontsize=fontsize + font_offset)
    plt.xlabel("Layers", fontsize=fontsize + font_offset)

    legend = plt.legend(fontsize=fontsize + font_offset, loc=legend_pos)
    legend.get_frame().set_alpha(0.5)  # Adjust alpha value as needed

    plt.grid(color="white", linewidth=4)

    plt.xticks(fontsize=fontsize + font_offset)
    plt.yticks(fontsize=fontsize + font_offset)

    if save_path is not None:
        save_plot_path = os.path.join(save_path, f"{plot_name}.jpg")
        plt.savefig(save_plot_path, bbox_inches="tight")
        print(f"saved at {save_plot_path}")

    plt.show()


def plt_avg_std_across_modalities_layers_sims(
    avg,
    std,
    xaxis,
    fontsize=25,
    font_offset=15,
    markersize=16,
    save_path=None,
    facecolor="lavender",
    plot_name="",
    across_epochs=False,
    epochs=[],
    y_label="Cosine Similarity",
    labels=[""],
    figsize=(20, 15),
    legend_pos="",
    plot_difference=False,
    skip_pp=False,
):

    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams["lines.linewidth"] = 4.0
    plt.rcParams["font.family"] = "serif"  # Times New Roman serif
    plt.figure(figsize=figsize)
    ax = plt.axes()
    ax.set_facecolor(facecolor)

    if not isinstance(avg, list):
        avg = [avg]
        std = [std]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    len_colors = len(colors) - 1
    j = 0

    for i, (avgs, stds, lab) in enumerate(zip(avg, std, labels)):

        avg_ = avgs
        std_ = stds

        if plot_difference:
            color = colors[i]
            averages = np.array([np.abs(a[0] - a[1]) for a in avg_])
            std_devs = np.array([np.sqrt(a[0] ** 2 + a[1] ** 2) for a in std_])

            plt.plot(
                xaxis,
                averages,
                marker="o",
                linestyle="--",
                markersize=markersize,
                label=f"{lab}-|PvsP - PvsT|",
                color=color,
            )
            plt.fill_between(xaxis, averages - std_devs, averages + std_devs, alpha=0.2)
            plt.axhline(y=np.mean(averages), linestyle="--", color=color)
        else:

            # Plot the data

            averages = np.array([a[0] for a in avg_])
            std_devs = np.array([a[0] for a in std_])
            xaxis = list(range(len(averages)))

            if not skip_pp:
                color = colors[j % len_colors]
                j += 1
                plt.plot(
                    xaxis,
                    averages,
                    marker="o",
                    linestyle="--",
                    markersize=markersize,
                    label=f"{lab}-P vs P",
                    color=color,
                )
                plt.fill_between(
                    xaxis, averages - std_devs, averages + std_devs, alpha=0.2
                )
                plt.axhline(y=np.mean(averages), linestyle="--", color=color)

            # Plot the data
            averages = np.array([a[1] for a in avg_])
            std_devs = np.array([a[1] for a in std_])

            color = colors[j % len_colors]
            j += 1
            plt.plot(
                xaxis,
                averages,
                marker="^",
                linestyle="--",
                markersize=markersize,
                label=f"{lab}-P vs T",
                color=color,
            )
            plt.fill_between(
                xaxis, averages - std_devs, averages + std_devs, alpha=0.15
            )
            plt.axhline(y=np.mean(averages), linestyle="--", color=color)

    plt.ylabel(y_label, fontsize=fontsize + font_offset)
    plt.xlabel("Layers", fontsize=fontsize + font_offset)

    legend = plt.legend(fontsize=fontsize + font_offset, loc=legend_pos)
    legend.get_frame().set_alpha(0.5)  # Adjust alpha value as needed

    plt.grid(color="white", linewidth=4)

    plt.xticks(fontsize=fontsize + font_offset)
    plt.yticks(fontsize=fontsize + font_offset)

    if save_path is not None:
        save_plot_path = os.path.join(save_path, f"{plot_name}.jpg")
        plt.savefig(save_plot_path, bbox_inches="tight")
        print(f"saved at {save_plot_path}")

    plt.show()


def make_it_symetric(img_txt_sims):
    img_txt_sims_ = []

    for i in range(len(img_txt_sims)):
        d, g, lab = img_txt_sims[i]
        n1, n2 = lab.split(" vs ")

        lab2 = f"{n2} vs {n1}"

        img_txt_sims_.append((d, g, lab))
        img_txt_sims_.append((d, g, lab2))

    # img_txt_sims_+= [([], 1, 'Text vs Text')]
    return img_txt_sims_


def filter_labels(data, remove_list=["Caption"]):
    data_ = []

    for i in range(len(data)):
        d, g, lab = data[i]
        for k in remove_list:
            lab = lab.replace(k, "")

        data_.append((d, g, lab))
    return data_


def compute_sims_from_hs_across_layers(
    all_embed_img,
    all_embed_txt,
    new_labels,
    layers=[0, 32],
    per_modality=False,
    unnormalized_sim=False,
    img_layer=None,
    sim_mode="normal",
    textlab="T",
    compute_intra=False,
):
    img_txt_sims = []

    all_avg = []
    all_std = []
    labels = []

    compute_sim_func = partial(
        compute_sim, unnormalized_sim=unnormalized_sim, mode=sim_mode
    )

    for layer in tqdm(layers):
        img_txt_sims = []

        if img_layer is None:
            img_layer_ = layer
        else:
            img_layer_ = img_layer

        for i in range(len(all_embed_img)):

            img = [im[img_layer_] for im in all_embed_img[i]]
            txt = [im[layer] for im in all_embed_txt[i]]
            labl = f"{new_labels[i]} vs {textlab}"
            img_txt_sims.append(compute_sim_func(img, txt) + (labl,))

        img_img_sims = []
        for i in range(len(all_embed_img)):
            for j in range(len(all_embed_img)):
                if i < j:
                    img = [im[img_layer_] for im in all_embed_img[i]]
                    txt = [im[layer] for im in all_embed_img[j]]

                    labl = f"{new_labels[i]} vs {new_labels[j]}"
                    img_img_sims.append(compute_sim_func(img, txt) + (labl,))

        txt_txt_sims = []
        for i in range(len(all_embed_txt)):
            for j in range(len(all_embed_txt)):
                if i < j:
                    img = [im[layer] for im in all_embed_txt[i]]
                    txt = [im[layer] for im in all_embed_txt[j]]

                    labl = f"{new_labels[i]}-{textlab} vs {new_labels[j]}-{textlab}"
                    txt_txt_sims.append(compute_sim_func(img, txt) + (labl,))

        img_img_intra_sims = []
        txt_txt_intra_sims = []
        if compute_intra:
            for i in range(len(all_embed_img)):
                size = len(all_embed_img[i]) // 2
                img = [im[layer] for im in all_embed_img[i][:size]]
                txt = [im[layer] for im in all_embed_img[i][size : 2 * size]]
                labl = f"{new_labels[i]} p vs p"
                img_img_intra_sims.append(compute_sim_func(img, txt) + (labl,))

            for i in range(len(all_embed_txt)):
                size = len(all_embed_txt[i]) // 2
                img = [im[layer] for im in all_embed_txt[i][:size]]
                txt = [im[layer] for im in all_embed_txt[i][size : 2 * size]]
                labl = f"{new_labels[i]} t vs t"
                txt_txt_intra_sims.append(compute_sim_func(img, txt) + (labl,))

        if per_modality:
            all_img_img = []
            all_img_txt = []
            all_txt_txt = []
            all_txt_txt_intra = []
            all_img_img_intra = []

            labels_img = []
            labels_txt = []
            labels_txt_txt = []
            labels_txt_txt_intra = []
            labels_img_img_intra = []

            for i in range(len(img_txt_sims)):
                all_img_txt.append(np.array(img_txt_sims[i][0]))
                labels_txt.append(img_txt_sims[i][-1])

            for i in range(len(img_img_sims)):
                all_img_img.append(np.array(img_img_sims[i][0]))
                labels_img.append(img_img_sims[i][-1])

            for i in range(len(txt_txt_sims)):
                all_txt_txt.append(np.array(txt_txt_sims[i][0]))
                labels_txt_txt.append(txt_txt_sims[i][-1])

            if compute_intra:
                for i in range(len(img_img_intra_sims)):
                    all_img_img_intra.append(np.array(img_img_intra_sims[i][0]))
                    labels_img_img_intra.append(img_img_intra_sims[i][-1])

                for i in range(len(txt_txt_intra_sims)):
                    all_txt_txt_intra.append(np.array(txt_txt_intra_sims[i][0]))
                    labels_txt_txt_intra.append(txt_txt_intra_sims[i][-1])

                all_avg.append(
                    [
                        [s.mean() for s in all_img_img],
                        [s.mean() for s in all_img_txt],
                        [s.mean() for s in all_txt_txt],
                        [s.mean() for s in all_img_img_intra],
                        [s.mean() for s in all_txt_txt_intra],
                    ]
                )
                all_std.append(
                    [
                        [s.std() for s in all_img_img],
                        [s.std() for s in all_img_txt],
                        [s.std() for s in all_txt_txt],
                        [s.std() for s in all_img_img_intra],
                        [s.std() for s in all_txt_txt_intra],
                    ]
                )
                labels = [
                    labels_img,
                    labels_txt,
                    labels_txt_txt,
                    labels_img_img_intra,
                    labels_txt_txt_intra,
                ]

            else:

                all_avg.append(
                    [
                        [s.mean() for s in all_img_img],
                        [s.mean() for s in all_img_txt],
                        [s.mean() for s in all_txt_txt],
                    ]
                )
                all_std.append(
                    [
                        [s.std() for s in all_img_img],
                        [s.std() for s in all_img_txt],
                        [s.std() for s in all_txt_txt],
                    ]
                )
                labels = [labels_img, labels_txt, labels_txt_txt]

        else:
            all_img_img = np.array([np.array(s[0]) for s in img_img_sims])
            all_img_txt = np.array([np.array(s[0]) for s in img_txt_sims])
            all_txt_txt = np.array([np.array(s[0]) for s in txt_txt_sims])

            if compute_intra:
                all_img_img_intra = np.array(
                    [np.array(s[0]) for s in img_img_intra_sims]
                )
                all_txt_txt_intra = np.array(
                    [np.array(s[0]) for s in txt_txt_intra_sims]
                )

                all_avg.append(
                    [
                        np.mean(all_img_img),
                        np.mean(all_img_txt),
                        np.mean(all_txt_txt),
                        np.mean(all_img_img_intra),
                        np.mean(all_txt_txt_intra),
                    ]
                )
                all_std.append(
                    [
                        np.std(all_img_img),
                        np.std(all_img_txt),
                        np.std(all_txt_txt),
                        np.std(all_img_img_intra),
                        np.std(all_txt_txt_intra),
                    ]
                )
            else:

                all_avg.append(
                    [np.mean(all_img_img), np.mean(all_img_txt), np.mean(all_txt_txt)]
                )
                all_std.append(
                    [np.std(all_img_img), np.std(all_img_txt), np.std(all_txt_txt)]
                )

    if per_modality:
        return all_avg, all_std, labels
    else:
        return all_avg, all_std


def plt_avg_std_sims_across(
    avgs,
    stds,
    xaxis,
    fontsize=30,
    font_offset=15,
    markersize=16,
    save_path=None,
    facecolor="lavender",
    plot_name="",
    labels=[],
    skip_pp=False,
    plot_tt=False,
    textlab="T",
    promptlab="P",
):

    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams["lines.linewidth"] = 4.0
    plt.rcParams["font.family"] = "serif"  # Times New Roman serif
    plt.figure(figsize=(20, 15))
    ax = plt.axes()
    ax.set_facecolor(facecolor)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_idx = 0

    for avg, std, lab in zip(avgs, stds, labels):

        averages = np.array([a[0] for a in avg])
        std_devs = np.array([a[0] for a in std])
        if not skip_pp:
            color = colors[color_idx]
            color_idx += 1
            if color_idx >= len(colors):
                color_idx = 0
            plt.plot(
                xaxis,
                averages,
                marker="o",
                linestyle="--",
                markersize=markersize,
                label=f"{promptlab} vs {promptlab} {lab}",
                color=color,
            )
            plt.fill_between(
                xaxis, averages - std_devs, averages + std_devs, alpha=0.2, color=color
            )
            plt.axhline(y=np.mean(averages), linestyle="--", zorder=3, color=color)

        averages = np.array([a[1] for a in avg])
        std_devs = np.array([a[1] for a in std])

        color = colors[color_idx]
        color_idx += 1
        if color_idx >= len(colors):
            color_idx = 0

        plt.plot(
            xaxis,
            averages,
            marker="^",
            linestyle="--",
            markersize=markersize,
            label=f"{promptlab} vs {textlab} {lab}",
            color=color,
        )
        plt.fill_between(
            xaxis, averages - std_devs, averages + std_devs, alpha=0.15, color=color
        )
        plt.axhline(y=np.mean(averages), linestyle="--", zorder=3, color=color)

        if plot_tt:

            color = colors[color_idx]
            color_idx += 1
            if color_idx >= len(colors):
                color_idx = 0
            averages = np.array([a[2] for a in avg])
            std_devs = np.array([a[2] for a in std])

            plt.plot(
                xaxis,
                averages,
                marker="*",
                linestyle="--",
                markersize=markersize,
                label=f"{textlab} vs {textlab} {lab}",
                color=color,
            )
            plt.fill_between(
                xaxis, averages - std_devs, averages + std_devs, alpha=0.2, color=color
            )
            plt.axhline(y=np.mean(averages), linestyle="--", zorder=3, color=color)

    plt.ylabel("Cosine Similarity", fontsize=fontsize + font_offset)
    plt.xlabel("Layers", fontsize=fontsize + font_offset)
    plt.legend(fontsize=fontsize + font_offset)

    plt.grid(color="white", linewidth=4)

    plt.xticks(fontsize=fontsize + font_offset)
    plt.yticks(fontsize=fontsize + font_offset)

    if save_path is not None:
        save_plot_path = os.path.join(save_path, f"{plot_name}.jpg")
        plt.savefig(save_plot_path, bbox_inches="tight")
        print(f"saved at {save_plot_path}")

    plt.show()


def plt_avg_std_sims(
    avg,
    std,
    xaxis,
    fontsize=30,
    font_offset=15,
    markersize=16,
    save_path=None,
    facecolor="lavender",
    plot_name="",
    labels=[],
    skip_pp=False,
    plot_tt=False,
    textlab="T",
    promptlab="P",
    plot_intra=False,
):

    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams["lines.linewidth"] = 4.0
    plt.rcParams["font.family"] = "serif"  # Times New Roman serif
    plt.figure(figsize=(20, 15))
    ax = plt.axes()
    ax.set_facecolor(facecolor)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_idx = 0

    averages = np.array([a[0] for a in avg])
    std_devs = np.array([a[0] for a in std])

    if not skip_pp:
        color = colors[color_idx]
        color_idx += 1
        if color_idx >= len(colors):
            color_idx = 0
        plt.plot(
            xaxis,
            averages,
            marker="o",
            linestyle="--",
            markersize=markersize,
            label=f"{promptlab} vs {promptlab}",
            color=color,
        )
        plt.fill_between(
            xaxis, averages - std_devs, averages + std_devs, alpha=0.2, color=color
        )
        plt.axhline(y=np.mean(averages), linestyle="--", color=color, zorder=3)

    averages = np.array([a[1] for a in avg])
    std_devs = np.array([a[1] for a in std])

    color = colors[color_idx]
    color_idx += 1
    if color_idx >= len(colors):
        color_idx = 0
    plt.plot(
        xaxis,
        averages,
        marker="^",
        linestyle="--",
        markersize=markersize,
        label=f"{promptlab} vs {textlab}",
        color=color,
    )
    plt.fill_between(
        xaxis, averages - std_devs, averages + std_devs, alpha=0.15, color=color
    )
    plt.axhline(y=np.mean(averages), linestyle="--", color=color, zorder=3)

    if plot_tt or plot_intra:
        color = colors[color_idx]
        color_idx += 1
        if color_idx >= len(colors):
            color_idx = 0
        averages = np.array([a[2] for a in avg])
        std_devs = np.array([a[2] for a in std])
        plt.plot(
            xaxis,
            averages,
            marker="o",
            linestyle="--",
            markersize=markersize,
            label=f"{textlab} vs {textlab}",
            color=color,
        )
        plt.fill_between(
            xaxis, averages - std_devs, averages + std_devs, alpha=0.2, color=color
        )
        plt.axhline(y=np.mean(averages), linestyle="--", color=color, zorder=3)

    if plot_intra:

        color = colors[color_idx]
        color_idx += 1
        if color_idx >= len(colors):
            color_idx = 0
        averages = np.array([a[3] for a in avg])
        std_devs = np.array([a[3] for a in std])
        plt.plot(
            xaxis,
            averages,
            marker="*",
            linestyle="--",
            markersize=markersize,
            label=f"p vs p",
            color=color,
        )
        plt.fill_between(
            xaxis, averages - std_devs, averages + std_devs, alpha=0.2, color=color
        )
        plt.axhline(y=np.mean(averages), linestyle="--", color=color, zorder=3)

        color = colors[color_idx]
        color_idx += 1
        if color_idx >= len(colors):
            color_idx = 0
        averages = np.array([a[4] for a in avg])
        std_devs = np.array([a[4] for a in std])
        plt.plot(
            xaxis,
            averages,
            marker="*",
            linestyle="--",
            markersize=markersize,
            label=f"t vs t",
            color=color,
        )
        plt.fill_between(
            xaxis, averages - std_devs, averages + std_devs, alpha=0.2, color=color
        )
        plt.axhline(y=np.mean(averages), linestyle="--", color=color, zorder=3)

    plt.ylabel("Cosine Similarity", fontsize=fontsize + font_offset)
    plt.xlabel("Layers", fontsize=fontsize + font_offset)
    plt.legend(fontsize=fontsize + font_offset)

    plt.grid(color="white", linewidth=4)

    plt.title("Embedding similarity")

    plt.xticks(fontsize=fontsize + font_offset)
    plt.yticks(fontsize=fontsize + font_offset)

    if save_path is not None:
        save_plot_path = os.path.join(save_path, f"{plot_name}.jpg")
        plt.savefig(save_plot_path, bbox_inches="tight")
        print(f"saved at {save_plot_path}")

    plt.show()


def plt_avg_std_sims_permodality(
    avg,
    std,
    xaxis,
    fontsize=30,
    font_offset=15,
    markersize=16,
    save_path=None,
    facecolor="lavender",
    plot_name="",
    labels=[],
    skip_pp=False,
    plot_tt=False,
    plot_intra=False,
):

    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams["lines.linewidth"] = 4.0
    plt.rcParams["font.family"] = "serif"  # Times New Roman serif
    plt.figure(figsize=(20, 15))
    ax = plt.axes()
    ax.set_facecolor(facecolor)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_idx = 0

    if not skip_pp:
        for i in range(len(avg[0][0])):

            color = colors[color_idx]
            color_idx += 1
            if color_idx >= len(colors):
                color_idx = 0
            averages = np.array([a[0][i] for a in avg])
            std_devs = np.array([a[0][i] for a in std])

            plt.plot(
                xaxis,
                averages,
                marker="o",
                linestyle="--",
                markersize=markersize,
                label=f"{labels[0][i]}",
                color=color,
            )
            plt.fill_between(
                xaxis, averages - std_devs, averages + std_devs, alpha=0.2, color=color
            )
            plt.axhline(y=np.mean(averages), linestyle="--", color=color, zorder=3)

    for i in range(len(avg[0][1])):

        color = colors[color_idx]
        color_idx += 1
        if color_idx >= len(colors):
            color_idx = 0
        averages = np.array([a[1][i] for a in avg])
        std_devs = np.array([a[1][i] for a in std])

        plt.plot(
            xaxis,
            averages,
            marker="^",
            linestyle="--",
            markersize=markersize,
            label=f"{labels[1][i]}",
            color=color,
        )
        plt.fill_between(
            xaxis, averages - std_devs, averages + std_devs, alpha=0.15, color=color
        )
        plt.axhline(y=np.mean(averages), linestyle="--", color=color, zorder=3)

    if plot_tt or plot_intra:
        for i in range(len(avg[0][2])):
            color = colors[color_idx]
            color_idx += 1
            if color_idx >= len(colors):
                color_idx = 0

            averages = np.array([a[2][i] for a in avg])
            std_devs = np.array([a[2][i] for a in std])

            plt.plot(
                xaxis,
                averages,
                marker="o",
                linestyle="--",
                markersize=markersize,
                label=f"{labels[2][i]}",
                color=color,
            )
            plt.fill_between(
                xaxis, averages - std_devs, averages + std_devs, alpha=0.2, color=color
            )
            plt.axhline(y=np.mean(averages), linestyle="--", color=color, zorder=3)

    if plot_intra:
        for i in range(len(avg[0][3])):
            color = colors[color_idx]
            color_idx += 1
            if color_idx >= len(colors):
                color_idx = 0

            averages = np.array([a[3][i] for a in avg])
            std_devs = np.array([a[3][i] for a in std])

            plt.plot(
                xaxis,
                averages,
                marker="*",
                linestyle="--",
                markersize=markersize,
                label=f"{labels[3][i]}",
                color=color,
            )
            plt.fill_between(
                xaxis, averages - std_devs, averages + std_devs, alpha=0.2, color=color
            )
            plt.axhline(y=np.mean(averages), linestyle="--", color=color, zorder=3)

            averages = np.array([a[4][i] for a in avg])
            std_devs = np.array([a[4][i] for a in std])

            plt.plot(
                xaxis,
                averages,
                marker="*",
                linestyle="--",
                markersize=markersize,
                label=f"{labels[4][i]}",
                color=color,
            )
            plt.fill_between(
                xaxis, averages - std_devs, averages + std_devs, alpha=0.2, color=color
            )
            plt.axhline(y=np.mean(averages), linestyle="--", color=color, zorder=3)

    plt.ylabel("Cosine Similarity", fontsize=fontsize + font_offset)
    plt.xlabel("Layers", fontsize=fontsize + font_offset)
    plt.legend(fontsize=fontsize + font_offset)

    plt.grid(color="white", linewidth=4)

    plt.title("Embedding similarity")

    plt.xticks(fontsize=fontsize + font_offset)
    plt.yticks(fontsize=fontsize + font_offset)

    if save_path is not None:
        save_plot_path = os.path.join(save_path, f"{plot_name}.jpg")
        plt.savefig(save_plot_path, bbox_inches="tight")
        print(f"saved at {save_plot_path}")

    plt.show()


results_dir = "/data/mshukor/logs/DePlam"
skip_last_txt_tokens = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="opt")
    parser.add_argument("--mode", type=str, default="inter")
    parser.add_argument("--sim_mode", type=str, default="normal")
    parser.add_argument("--file_name", type=str, default="all_hidden_states.pth")
    parser.add_argument("--noise_baseline", action="store_true")

    args = parser.parse_args()

    suffix = ""

    labels = [
        "V",
        "I",
        "A",
    ]

    print(f"read LLM for {args.model} ...")

    prompt_len = 10

    if "opt" in args.model and "llava" not in args.model:

        save_results_dir = "/data/mshukor/logs/DePlam/results/plots/opt"
        folders = [
            "ePALM_L_video_caption_l0_1_qsformerl10_timesformer",
            "ePALM_caption_qformerl10_vitl",
            "ePALM_L_audio_caption_l0_1_qsformerl10_ast",
        ]

    elif "vicuna" in args.model:

        save_results_dir = "/data/mshukor/logs/DePlam/results/plots/vicuna"
        folders = [
            "ePALM_L_video_caption_l0_1_qsformerl10_llamav2vicuna_timesformer",
            "ePALM_caption_qformerl10_llamav2vicuna_vitl",
            "ePALM_L_audio_caption_l0_1_qsformerl10_llamav2vicuna_ast",
        ]

        if args.noise_baseline:
            folders.append("ePALM_caption_noise_qformerl10_llamav2vicuna_vitl")
            labels.append("N")

    elif args.model == "llamav2":
        save_results_dir = "/data/mshukor/logs/DePlam/results/plots/llamav2"
        folders = [
            "ePALM_L_video_caption_l0_1_qsformerl10_llamav2_timesformer",
            "ePALM_caption_qformerl10_llamav2_vitl",
            "ePALM_L_audio_caption_l0_1_qsformerl10_llamav2_ast",
        ]

    elif "llava" in args.model:

        text_model = "lmsys/vicuna-7b-v1.5"
        print(f"Reading {text_model} ...")

        save_results_dir = "/data/mshukor/logs/DePlam/results/plots/llava"
        prompt_len = 576

        main_results_dir = "/data/mshukor/logs/DePlam/llava/"

        if "qformernoptllavafrozen1round" in args.model:

            results_dir = "/data/mshukor/logs/DePlam/llava/llava_v1_5_qformer/hidden_states_1round/"
            mdl = "qformernoptllavafrozen1round"
            prompt_len = 32

        elif "qformertextcondnoptllavafrozen1round" in args.model:

            results_dir = "/data/mshukor/logs/DePlam/llava/llava_v1_5_qformertextcondfix1_l1_onlyimgtxt_2layers1024/checkpoint-6000/hidden_states_1round/"
            mdl = "qformertextcondnoptllavafrozen1round"

        elif "nopt_ckpts_llavafrozen1round" in args.model:

            results_dir = "/data/mshukor/logs/DePlam/llava/llava_v1_5_baseline_nopt/"
            mdl = "nopt_ckpts_llavafrozen1round"

        elif "noptllavafrozen1round" in args.model:

            results_dir = "/data/mshukor/logs/DePlam/llava/llava_v1_5_baseline_v100/hidden_states_1round/"
            mdl = "noptllavafrozen1round"

        elif "llavafrozen1round" in args.model:
            results_dir = "/data/mshukor/logs/DePlam/llava/llava_v1_5_baseline_withpt/hidden_states_1round/"
            mdl = "llavafrozen1round"

        elif "llava1round" in args.model:
            results_dir = (
                "/data/mshukor/logs/DePlam/llava/llava-v1.5-7b/hidden_states_1round/"
            )
            mdl = "llava1round"

        else:
            mdl = "allllava"

        if "alltok" in args.mode and "inside" in args.mode:
            results_dir = results_dir.replace(
                "hidden_states_1round", "hidden_states_1round_atts"
            )

        all_models = {
            "qformernoptllavafrozen1round": "/data/mshukor/logs/DePlam/llava/llava_v1_5_qformer/hidden_states_1round/",
            "noptllavafrozen1round": "/data/mshukor/logs/DePlam/llava/llava_v1_5_baseline_v100/hidden_states_1round/",
            "llavafrozen1round": "/data/mshukor/logs/DePlam/llava/llava_v1_5_baseline_withpt/hidden_states_1round/",
            "llava1round": "/data/mshukor/logs/DePlam/llava/llava-v1.5-7b/hidden_states_1round/",
        }
        modelname_2_label = {
            "llava1round": "LLaVA-v1.5-1",
            "llavafrozen1round": "LLaVA-v1.5-2",
            "noptllavafrozen1round": "LLaVA-v1.5-3",
            "qformernoptllavafrozen1round": "LLaVA-v1.5-4",
        }

        modelname_2_prompt = {
            "llava1round": 576,
            "llavafrozen1round": 576,
            "noptllavafrozen1round": 576,
            "qformernoptllavafrozen1round": 32,
        }

        folders = [
            "aokvqa",
            "caption",
            "gqa",
            "vqav2",
        ]

        labels = [
            "A-OKVQA I",
            "COCO Caption I",
            "GQA I",
            "VQAv2 I",
        ]

    else:
        raise NotImplemented

    if args.noise_baseline:
        suffix += "_noise"

    new_labels = labels

    if any([k in args.sim_mode for k in ["mean", "max", "median", "min"]]):
        index_img = -1
        index_txt = -1
        suffix += f"{args.sim_mode}_sim"
    else:
        index_img = None
        index_txt = None

    # intervals = [1, 2, 4, 5, 8]
    intervals = [1, 8]
    # intervals = [1, 8]

    if "inside" in args.mode:
        hidden_states_key = [
            "intermediate_hidden_preatts",
            "intermediate_hidden_atts",
            "intermediate_hidden_atts_res",
            "intermediate_hidden_prefc",
            "intermediate_hidden_act",
            "intermediate_hidden_fc2",
            "intermediate_hidden_fc2_res",
        ]

        hskey_2_hsname = {
            "intermediate_hidden_preatts": "||LN1||",
            "intermediate_hidden_atts": "||SA||",
            "intermediate_hidden_atts_res": "||X + SA||",
            "intermediate_hidden_prefc": "||LN2||",
            "intermediate_hidden_act": "||FC1||",
            "intermediate_hidden_fc2": "||FC2||",
            "intermediate_hidden_fc2_res": "||X + FC2||",
        }

        # hidden_states_key = ['intermediate_hidden_atts']

        if "alltok" not in args.mode:
            prompt_len = 1  # we don't store all tokens inside
        if "alltok" in args.mode:
            suffix += "_alltok"
            hidden_states_key = ["intermediate_hidden_atts"]

    else:
        hidden_states_key = ["hidden_states"]

    if "llava" in args.model:

        if "system" in args.mode:
            discard_first_text = False
            suffix += "_withsystemmsg"
        else:
            discard_first_text = True

        file_name = args.file_name
        if "1k" in file_name:
            suffix += "_1k"

        if "imagelayer0" in args.mode:
            img_layer = 0
            suffix += "_imgl0"
        else:
            img_layer = None

        print(args.mode, suffix)
        if "epochs" in args.mode:
            avgs = []
            stds = []

            if "allepochs" in args.mode:
                suffix += "_allepochs"
                epochs = ["0000", "1500", "3000", "4500", "6000", "7500"]
            else:
                epochs = ["0000", "3000", "4500", "6000"]

            if "skipfirst" in args.mode:
                epochs = epochs[1:]

                suffix += "_skipfirst"

            if "task" in args.mode:
                for i, folder_ in enumerate(folders):
                    folder_ = [folder_]
                    labels_ = [labels[i]]

                    for e in epochs:

                        results_dir_ = os.path.join(
                            results_dir, f"checkpoint-{e}", "hidden_states_1round"
                        )

                        print(f"epoch: {e}", results_dir_)

                        file_name = f"all_hidden_states.pth"

                        plot_suffix = file_name.split(".")[0]
                        (
                            all_embed_txt,
                            all_embed_img,
                            new_labels,
                        ) = read_hidden_states_from_folders_llava(
                            results_dir_,
                            folder_,
                            file_name,
                            labels_,
                            index_img,
                            index_txt,
                            prompt_len=prompt_len,
                            discard_first_text=discard_first_text,
                            hidden_states_key=hidden_states_key,
                        )

                        suffix_ = suffix + f"epoch{e}" + f"{labels_[0]}"
                        for i, hskey in enumerate(hidden_states_key):

                            print(hskey)
                            if hskey != "hidden_states":
                                suffix_ = suffix + f"_{hskey}_"

                            if len(hidden_states_key) > 1:
                                all_embed_txt_ = [embds[i] for embds in all_embed_txt]
                                all_embed_img_ = [embds[i] for embds in all_embed_img]
                            else:
                                all_embed_txt_ = all_embed_txt
                                all_embed_img_ = all_embed_img

                            plot_layers = list(range(32))
                            all_avg, all_std = compute_sims_from_hs_across_layers(
                                all_embed_img_,
                                all_embed_txt_,
                                new_labels,
                                plot_layers,
                                sim_mode=args.sim_mode,
                                img_layer=img_layer,
                            )

                            avgs.append(all_avg)
                            stds.append(all_std)

                            plt_avg_std_sims(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{mdl}_emb_sim_layers_avg{suffix_}",
                                save_path=save_results_dir,
                                fontsize=30,
                            )
                            plt_avg_std_sims(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{mdl}_emb_sim_layers_avg_skip_pp{suffix_}",
                                save_path=save_results_dir,
                                fontsize=30,
                                skip_pp=True,
                            )
                            plt_avg_std_sims(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{mdl}_emb_sim_layers_avg_plot_tt{suffix_}",
                                save_path=save_results_dir,
                                fontsize=30,
                                plot_tt=True,
                            )

                    epochs_ = [f"step {s}" for s in epochs]

                    plt_avg_std_sims_across(
                        avgs,
                        stds,
                        plot_layers,
                        plot_name=f"{mdl}_emb_sim_layers_avg_all_epochs{suffix}",
                        save_path=save_results_dir,
                        labels=epochs_,
                    )
                    plt_avg_std_sims_across(
                        avgs,
                        stds,
                        plot_layers,
                        plot_name=f"{mdl}_emb_sim_layers_avg_all_epochs_skip_pp{suffix}",
                        save_path=save_results_dir,
                        labels=epochs_,
                        skip_pp=True,
                    )
                    plt_avg_std_sims_across(
                        avgs,
                        stds,
                        plot_layers,
                        plot_name=f"{mdl}_emb_sim_layers_avg_all_epochs_plot_tt{suffix}",
                        save_path=save_results_dir,
                        labels=epochs_,
                        plot_tt=True,
                    )

            else:
                for e in epochs:

                    results_dir_ = os.path.join(
                        results_dir, f"checkpoint-{e}", "hidden_states_1round"
                    )

                    print(f"epoch: {e}", results_dir_)

                    file_name = f"all_hidden_states.pth"

                    plot_suffix = file_name.split(".")[0]
                    (
                        all_embed_txt,
                        all_embed_img,
                        new_labels,
                    ) = read_hidden_states_from_folders_llava(
                        results_dir_,
                        folders,
                        file_name,
                        labels,
                        index_img,
                        index_txt,
                        prompt_len=prompt_len,
                        discard_first_text=discard_first_text,
                        hidden_states_key=hidden_states_key,
                    )

                    suffix_ = suffix + f"epoch{e}"
                    for i, hskey in enumerate(hidden_states_key):

                        print(hskey)
                        if hskey != "hidden_states":
                            suffix_ = suffix + f"_{hskey}_"

                        if len(hidden_states_key) > 1:
                            all_embed_txt_ = [embds[i] for embds in all_embed_txt]
                            all_embed_img_ = [embds[i] for embds in all_embed_img]
                        else:
                            all_embed_txt_ = all_embed_txt
                            all_embed_img_ = all_embed_img

                        plot_layers = list(range(32))
                        if "consecutive" in args.mode:
                            font_offset = 18
                            figsize = (35, 30)
                            legend_pos = "lower right"
                            labels_to_remove = []

                            ### norms
                            (
                                all_avg,
                                all_std,
                            ) = compute_norm_from_hs_across_consecutive_layers(
                                all_embed_img,
                                all_embed_txt,
                                plot_layers,
                                per_modality=False,
                                sim_mode=args.sim_mode,
                            )

                            plt_avg_std_consecutive_layers_sims_llava(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{mdl}_emb_norm_consecutive_layers_avg{suffix_}",
                                save_path=save_results_dir,
                                y_label="Tokens Norm",
                                labels=[""],
                                legend_pos=legend_pos,
                                figsize=figsize,
                                labels_to_remove=labels_to_remove,
                            )
                            avgs.append(all_avg)
                            stds.append(all_std)

                            ### norms
                            (
                                all_avg,
                                all_std,
                            ) = compute_norm_from_hs_across_consecutive_layers(
                                all_embed_img,
                                all_embed_txt,
                                plot_layers,
                                per_modality=True,
                                sim_mode=args.sim_mode,
                            )
                            all_avg_, all_std_ = organise_per_modal_dict(
                                all_avg, all_std
                            )

                            plt_avg_std_consecutive_layers_sims_llava(
                                all_avg_,
                                all_std_,
                                plot_layers,
                                plot_name=f"{mdl}_emb_norm_consecutive_layers_avg_permodal{suffix_}",
                                save_path=save_results_dir,
                                y_label="Tokens Norm",
                                labels=labels,
                                legend_pos=legend_pos,
                                figsize=figsize,
                                labels_to_remove=labels_to_remove,
                            )

                        else:
                            all_avg, all_std = compute_sims_from_hs_across_layers(
                                all_embed_img_,
                                all_embed_txt_,
                                new_labels,
                                plot_layers,
                                sim_mode=args.sim_mode,
                                img_layer=img_layer,
                            )

                            avgs.append(all_avg)
                            stds.append(all_std)

                            plt_avg_std_sims(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{mdl}_emb_sim_layers_avg{suffix_}",
                                save_path=save_results_dir,
                                fontsize=30,
                            )
                            plt_avg_std_sims(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{mdl}_emb_sim_layers_avg_skip_pp{suffix_}",
                                save_path=save_results_dir,
                                fontsize=30,
                                skip_pp=True,
                            )
                            plt_avg_std_sims(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{mdl}_emb_sim_layers_avg_plot_tt{suffix_}",
                                save_path=save_results_dir,
                                fontsize=30,
                                plot_tt=True,
                            )

                            (
                                all_avg,
                                all_std,
                                labels_,
                            ) = compute_sims_from_hs_across_layers(
                                all_embed_img_,
                                all_embed_txt_,
                                new_labels,
                                plot_layers,
                                per_modality=True,
                                unnormalized_sim=False,
                                sim_mode=args.sim_mode,
                                img_layer=img_layer,
                            )
                            # labels_ = [labels, labels, labels]
                            plt_avg_std_sims_permodality(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{mdl}_emb_sim_layers_avg_permodal{suffix_}",
                                save_path=save_results_dir,
                                labels=labels_,
                            )
                            plt_avg_std_sims_permodality(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{mdl}_emb_sim_layers_avg_permodal_skip_pp{suffix_}",
                                save_path=save_results_dir,
                                labels=labels_,
                                skip_pp=True,
                            )
                            plt_avg_std_sims_permodality(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{mdl}_emb_sim_layers_avg_permodal_plot_tt{suffix_}",
                                save_path=save_results_dir,
                                labels=labels_,
                                plot_tt=True,
                            )

                epochs_ = [f"step {s}" for s in epochs]

                if "consecutive" in args.mode:
                    plt_avg_std_consecutive_layers_sims_across(
                        avgs,
                        stds,
                        plot_layers,
                        plot_name=f"{suffix}emb_norm_layers_avg_all_epochs",
                        save_path=save_results_dir,
                        metalabels=epochs_,
                        labels=labels,
                    )

                else:
                    plt_avg_std_sims_across(
                        avgs,
                        stds,
                        plot_layers,
                        plot_name=f"{mdl}_emb_sim_layers_avg_all_epochs{suffix}",
                        save_path=save_results_dir,
                        labels=epochs_,
                    )
                    plt_avg_std_sims_across(
                        avgs,
                        stds,
                        plot_layers,
                        plot_name=f"{mdl}_emb_sim_layers_avg_all_epochs_skip_pp{suffix}",
                        save_path=save_results_dir,
                        labels=epochs_,
                        skip_pp=True,
                    )
                    plt_avg_std_sims_across(
                        avgs,
                        stds,
                        plot_layers,
                        plot_name=f"{mdl}_emb_sim_layers_avg_all_epochs_plot_tt{suffix}",
                        save_path=save_results_dir,
                        labels=epochs_,
                        plot_tt=True,
                    )

        elif "models" in args.mode:

            avgs = []
            stds = []

            mdls = []
            for mdl in all_models.keys():

                mdls.append(modelname_2_label[mdl])
                results_dir_ = all_models[mdl]
                prompt_len_ = modelname_2_prompt[mdl]

                print(f"model: {mdl}", results_dir_)

                file_name = f"all_hidden_states.pth"

                (
                    all_embed_txt,
                    all_embed_img,
                    new_labels,
                ) = read_hidden_states_from_folders_llava(
                    results_dir_,
                    folders,
                    file_name,
                    labels,
                    index_img,
                    index_txt,
                    prompt_len=prompt_len_,
                    discard_first_text=discard_first_text,
                    hidden_states_key=hidden_states_key,
                )

                suffix_ = suffix + f"model_{mdl}"
                for i, hskey in enumerate(hidden_states_key):

                    print(hskey)
                    if hskey != "hidden_states":
                        suffix_ = suffix + f"_{hskey}_"

                    if len(hidden_states_key) > 1:
                        all_embed_txt_ = [embds[i] for embds in all_embed_txt]
                        all_embed_img_ = [embds[i] for embds in all_embed_img]
                    else:
                        all_embed_txt_ = all_embed_txt
                        all_embed_img_ = all_embed_img

                    plot_layers = list(range(32))
                    all_avg, all_std = compute_sims_from_hs_across_layers(
                        all_embed_img_,
                        all_embed_txt_,
                        new_labels,
                        plot_layers,
                        sim_mode=args.sim_mode,
                        img_layer=img_layer,
                    )

                    avgs.append(all_avg)
                    stds.append(all_std)

                    plt_avg_std_sims(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{mdl}_emb_sim_layers_avg{suffix_}",
                        save_path=save_results_dir,
                        fontsize=30,
                    )
                    plt_avg_std_sims(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{mdl}_emb_sim_layers_avg_skip_pp{suffix_}",
                        save_path=save_results_dir,
                        fontsize=30,
                        skip_pp=True,
                    )
                    plt_avg_std_sims(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{mdl}_emb_sim_layers_avg_plot_tt{suffix_}",
                        save_path=save_results_dir,
                        fontsize=30,
                        plot_tt=True,
                    )

            epochs_ = [f"{mdl}" for mdl in mdls]

            plt_avg_std_sims_across(
                avgs,
                stds,
                plot_layers,
                plot_name=f"allmodels_emb_sim_layers_avg_all_epochs{suffix}",
                save_path=save_results_dir,
                labels=epochs_,
            )
            plt_avg_std_sims_across(
                avgs,
                stds,
                plot_layers,
                plot_name=f"allmodels_emb_sim_layers_avg_all_epochs_skip_pp{suffix}",
                save_path=save_results_dir,
                labels=epochs_,
                skip_pp=True,
            )
            plt_avg_std_sims_across(
                avgs,
                stds,
                plot_layers,
                plot_name=f"allmodels_emb_sim_layers_avg_all_epochs_plot_tt{suffix}",
                save_path=save_results_dir,
                labels=epochs_,
                plot_tt=True,
            )

        else:
            plot_suffix = file_name.split(".")[0]
            (
                all_embed_txt,
                all_embed_img,
                new_labels,
            ) = read_hidden_states_from_folders_llava(
                results_dir,
                folders,
                file_name,
                labels,
                index_img,
                index_txt,
                prompt_len=prompt_len,
                discard_first_text=discard_first_text,
                hidden_states_key=hidden_states_key,
            )

            if "inside" in args.mode:
                plot_layers = list(range(1, 33))
            else:
                plot_layers = list(range(32))

            all_avg_sims = []
            all_std_sims = []

            all_avg_norm = []
            all_std_norm = []

            if "consecutive" not in args.mode:
                print("non consecutive llava ...")
                if "intra" in args.mode:
                    compute_intra = True
                else:
                    compute_intra = False

                suffix_ = suffix

                for i, hskey in enumerate(hidden_states_key):

                    print(hskey)
                    if hskey != "hidden_states":
                        suffix_ = suffix + f"_{hskey}_"

                    if len(hidden_states_key) > 1:
                        all_embed_txt_ = [embds[i] for embds in all_embed_txt]
                        all_embed_img_ = [embds[i] for embds in all_embed_img]
                    else:
                        all_embed_txt_ = all_embed_txt
                        all_embed_img_ = all_embed_img

                    all_avg, all_std, labels_ = compute_sims_from_hs_across_layers(
                        all_embed_img_,
                        all_embed_txt_,
                        new_labels,
                        plot_layers,
                        per_modality=True,
                        sim_mode=args.sim_mode,
                        compute_intra=compute_intra,
                    )

                    plt_avg_std_sims_permodality(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{mdl}_emb_sim_layers_avg_permodal{suffix_}",
                        save_path=save_results_dir,
                        labels=labels_,
                    )
                    plt_avg_std_sims_permodality(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{mdl}_emb_sim_layers_avg_permodal_skip_pp{suffix_}",
                        save_path=save_results_dir,
                        labels=labels_,
                        skip_pp=True,
                    )
                    plt_avg_std_sims_permodality(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{mdl}_emb_sim_layers_avg_permodal_plot_tt{suffix_}",
                        save_path=save_results_dir,
                        labels=labels_,
                        skip_pp=True,
                        plot_tt=True,
                    )

                    if compute_intra:
                        plt_avg_std_sims_permodality(
                            all_avg,
                            all_std,
                            plot_layers,
                            plot_name=f"{mdl}_emb_sim_layers_avg_permodal_plot_tt_pp{suffix_}",
                            save_path=save_results_dir,
                            labels=labels_,
                            skip_pp=True,
                            plot_intra=True,
                        )

                    all_avg, all_std = compute_sims_from_hs_across_layers(
                        all_embed_img_,
                        all_embed_txt_,
                        new_labels,
                        plot_layers,
                        per_modality=False,
                        sim_mode=args.sim_mode,
                        compute_intra=compute_intra,
                    )

                    all_avg_sims.append(all_avg)
                    all_std_sims.append(all_std)

                    plt_avg_std_sims(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{mdl}_emb_sim_layers_avg{suffix_}",
                        save_path=save_results_dir,
                        labels=[""],
                    )
                    plt_avg_std_sims(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{mdl}_emb_sim_layers_avg_skip_pp{suffix_}",
                        save_path=save_results_dir,
                        labels=[""],
                        skip_pp=True,
                    )
                    plt_avg_std_sims(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{mdl}_emb_sim_layers_avg_plot_tt{suffix_}",
                        save_path=save_results_dir,
                        labels=[""],
                        plot_tt=True,
                    )

                    if compute_intra:
                        plt_avg_std_sims(
                            all_avg,
                            all_std,
                            plot_layers,
                            plot_name=f"{mdl}_emb_sim_layers_avg_plot_tt_pp{suffix_}",
                            save_path=save_results_dir,
                            labels=[""],
                            plot_intra=True,
                        )

                if "inside" in args.mode:

                    key_labels = [hskey_2_hsname[hskey] for hskey in hidden_states_key]
                    # key_labels = ['||LN1||', '||SA||', '||X + SA||', '||LN2||', '||FC1||', '||FC2||', '||X + FC2||']
                    ids_to_keep = [0, 1, 3, 4, 5]

                    key_labels_ = [key_labels[i] for i in ids_to_keep]
                    all_avg_sims_ = [all_avg_sims[i] for i in ids_to_keep]
                    all_std_sims_ = [all_std_sims[i] for i in ids_to_keep]

                    legend_pos = "lower right"

                    figsize = (30, 25)
                    font_offset = 15
                    plot_layers = list(range(32))
                    plt_avg_std_across_modalities_layers_sims(
                        all_avg_sims_,
                        all_std_sims_,
                        plot_layers,
                        plot_name=f"{mdl}_{suffix}emb_sim_all_inside_layers_avg",
                        save_path=save_results_dir,
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                    )

                    plt_avg_std_across_modalities_layers_sims(
                        all_avg_sims_,
                        all_std_sims_,
                        plot_layers,
                        plot_name=f"{mdl}_{suffix}emb_sim_all_inside_layers_avg_skip_pp",
                        save_path=save_results_dir,
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                        skip_pp=True,
                    )

                    plt_avg_std_across_modalities_layers_sims(
                        all_avg_sims_,
                        all_std_sims_,
                        plot_layers,
                        plot_name=f"{mdl}_{suffix}emb_diff_sim_all_inside_layers_avg",
                        save_path=save_results_dir,
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                        plot_difference=True,
                    )

                    ids_to_keep = [0, 1, 2, 3, 4, 5, 6]

                    key_labels_ = [key_labels[i] for i in ids_to_keep]
                    all_avg_sims_ = [all_avg_sims[i] for i in ids_to_keep]
                    all_std_sims_ = [all_std_sims[i] for i in ids_to_keep]

                    plt_avg_std_across_modalities_layers_sims(
                        all_avg_sims_,
                        all_std_sims_,
                        plot_layers,
                        plot_name=f"{mdl}_{suffix}emb_sim_all_inside_layers_avg_allin",
                        save_path=save_results_dir,
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                    )

                    plt_avg_std_across_modalities_layers_sims(
                        all_avg_sims_,
                        all_std_sims_,
                        plot_layers,
                        plot_name=f"{mdl}_{suffix}emb_sim_all_inside_layers_avg_skip_pp_allin",
                        save_path=save_results_dir,
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                        skip_pp=True,
                    )

                    plt_avg_std_across_modalities_layers_sims(
                        all_avg_sims_,
                        all_std_sims_,
                        plot_layers,
                        plot_name=f"{mdl}_{suffix}emb_diff_sim_all_inside_layers_avg_allin",
                        save_path=save_results_dir,
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                        plot_difference=True,
                    )

            else:
                print("consecutive llava ...")

                suffix_ = suffix

                if "inside" in args.mode:
                    intervals = [1]

                for i, hskey in enumerate(hidden_states_key):

                    print(hskey)
                    if hskey != "hidden_states" or (
                        hskey == "hidden_states" and "inside" in args.mode
                    ):
                        suffix_ = suffix + f"_{hskey}_"

                    if len(hidden_states_key) > 1:
                        all_embed_txt_ = [embds[i] for embds in all_embed_txt]
                        all_embed_img_ = [embds[i] for embds in all_embed_img]
                    else:
                        all_embed_txt_ = all_embed_txt
                        all_embed_img_ = all_embed_img

                    all_avg, all_std = compute_sims_from_hs_across_consecutive_layers(
                        all_embed_img_,
                        all_embed_txt_,
                        plot_layers,
                        intervals,
                        per_modality=True,
                        sim_mode=args.sim_mode,
                    )
                    all_avg_, all_std_ = organise_per_modal_dict(all_avg, all_std)

                    font_offset = 18
                    figsize = (35, 30)
                    legend_pos = "lower right"
                    labels_to_remove = (
                        []
                    )  # ['OKVQA', 'RefCOCO', 'GQA', 'OCR-VQA', 'Text VQA']

                    plt_avg_std_consecutive_layers_sims_llava(
                        all_avg_,
                        all_std_,
                        plot_layers,
                        plot_name=f"{mdl}_emb_sim_consecutive_layers_avg_permodal{suffix_}",
                        save_path=save_results_dir,
                        labels=labels,
                        legend_pos=legend_pos,
                        figsize=figsize,
                        labels_to_remove=labels_to_remove,
                    )

                    ### norms
                    all_avg, all_std = compute_norm_from_hs_across_consecutive_layers(
                        all_embed_img_,
                        all_embed_txt_,
                        plot_layers,
                        per_modality=True,
                        sim_mode=args.sim_mode,
                    )
                    all_avg_, all_std_ = organise_per_modal_dict(all_avg, all_std)

                    plt_avg_std_consecutive_layers_sims_llava(
                        all_avg_,
                        all_std_,
                        plot_layers,
                        plot_name=f"{mdl}_emb_norm_consecutive_layers_avg_permodal{suffix_}",
                        save_path=save_results_dir,
                        y_label="Tokens Norm",
                        labels=labels,
                        legend_pos=legend_pos,
                        figsize=figsize,
                        labels_to_remove=labels_to_remove,
                    )

                    all_avg, all_std = compute_sims_from_hs_across_consecutive_layers(
                        all_embed_img_,
                        all_embed_txt_,
                        plot_layers,
                        intervals,
                        per_modality=False,
                        sim_mode=args.sim_mode,
                    )
                    font_offset = 18
                    figsize = (25, 20)
                    legend_pos = "lower right"
                    labels_to_remove = (
                        []
                    )  # ['OKVQA', 'RefCOCO', 'GQA', 'OCR-VQA', 'Text VQA']

                    plt_avg_std_consecutive_layers_sims_llava(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{mdl}_emb_sim_consecutive_layers_avg{suffix_}",
                        save_path=save_results_dir,
                        labels=[""],
                        legend_pos=legend_pos,
                        figsize=figsize,
                        labels_to_remove=labels_to_remove,
                    )

                    all_avg_sims.append(all_avg)
                    all_std_sims.append(all_std)

                    ### norms
                    all_avg, all_std = compute_norm_from_hs_across_consecutive_layers(
                        all_embed_img_,
                        all_embed_txt_,
                        plot_layers,
                        per_modality=False,
                        sim_mode=args.sim_mode,
                    )

                    plt_avg_std_consecutive_layers_sims_llava(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{mdl}_emb_norm_consecutive_layers_avg{suffix_}",
                        save_path=save_results_dir,
                        y_label="Tokens Norm",
                        labels=[""],
                        legend_pos=legend_pos,
                        figsize=figsize,
                        labels_to_remove=labels_to_remove,
                    )

                    all_avg_norm.append(all_avg)
                    all_std_norm.append(all_std)

                if "inside" in args.mode:
                    key_labels = [hskey_2_hsname[hskey] for hskey in hidden_states_key]

                    ids_to_keep = [0, 1, 3, 4, 5]

                    key_labels_ = [key_labels[i] for i in ids_to_keep]
                    all_avg_sims_ = [all_avg_sims[i] for i in ids_to_keep]
                    all_std_sims_ = [all_std_sims[i] for i in ids_to_keep]

                    all_avg_norm_ = [all_avg_norm[i] for i in ids_to_keep]
                    all_std_norm_ = [all_std_norm[i] for i in ids_to_keep]

                    legend_pos = "lower right"

                    figsize = (30, 25)
                    font_offset = 15
                    plot_layers = list(range(1, 33))
                    plt_avg_std_consecutive_layers_sims(
                        all_avg_sims_,
                        all_std_sims_,
                        plot_layers,
                        plot_name=f"{mdl}_{suffix}emb_sim_all_inside_consecutive_layers_avg",
                        save_path=save_results_dir,
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                    )

                    plot_layers = list(range(1, 33))
                    plt_avg_std_consecutive_layers_sims(
                        all_avg_norm_,
                        all_std_norm_,
                        plot_layers,
                        plot_name=f"{mdl}_{suffix}emb_norm_all_inside_consecutive_layers_avg",
                        save_path=save_results_dir,
                        y_label="Tokens Norm",
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                    )

                    ids_to_keep = [0, 1, 2, 3, 4, 5, 6]
                    key_labels_ = [key_labels[i] for i in ids_to_keep]
                    all_avg_sims_ = [all_avg_sims[i] for i in ids_to_keep]
                    all_std_sims_ = [all_std_sims[i] for i in ids_to_keep]

                    all_avg_norm_ = [all_avg_norm[i] for i in ids_to_keep]
                    all_std_norm_ = [all_std_norm[i] for i in ids_to_keep]

                    legend_pos = "lower right"

                    figsize = (30, 25)
                    font_offset = 15
                    plot_layers = list(range(1, 33))
                    plt_avg_std_consecutive_layers_sims(
                        all_avg_sims_,
                        all_std_sims_,
                        plot_layers,
                        plot_name=f"{mdl}_{suffix}emb_sim_all_inside_consecutive_layers_avg_allin",
                        save_path=save_results_dir,
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                    )

                    plot_layers = list(range(1, 33))
                    plt_avg_std_consecutive_layers_sims(
                        all_avg_norm_,
                        all_std_norm_,
                        plot_layers,
                        plot_name=f"{mdl}_{suffix}emb_norm_all_inside_consecutive_layers_avg_allin",
                        save_path=save_results_dir,
                        y_label="Tokens Norm",
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                    )

    else:

        file_name = args.file_name
        if "1k" in file_name:
            suffix += "_1k"

        if "imagelayer0" in args.mode:
            img_layer = 0
            suffix += "_imgl0"
        else:
            img_layer = None

        if "epochs_consecutive" in args.mode:
            suffix += "_epochs_consecutive"
            avgs1 = []
            stds1 = []

            avgs2 = []
            stds2 = []

            epochs = [-1, 0, 1, 2]
            for i in range(len(epochs) - 1):
                e1 = epochs[i]
                e2 = epochs[i + 1]

                print(f"epochs: {e1} {e2}")

                if e1 >= 0:
                    file_name = f"checkpoint_{e1}all_hidden_states.pth"
                else:
                    file_name = f"scratchall_hidden_states.pth"
                plot_suffix = file_name.split(".")[0]
                (
                    all_embed_txt1,
                    all_embed_img1,
                    new_labels,
                    all_embed_img_before,
                    all_embed_img_connector,
                ) = read_hidden_states_from_folders(
                    results_dir,
                    folders,
                    file_name,
                    labels,
                    index_img,
                    index_txt,
                    prompt_len=prompt_len,
                    hidden_states_key=hidden_states_key,
                )

                file_name = f"checkpoint_{e2}all_hidden_states.pth"
                plot_suffix = file_name.split(".")[0]
                (
                    all_embed_txt2,
                    all_embed_img2,
                    new_labels,
                    all_embed_img_before,
                    all_embed_img_connector,
                ) = read_hidden_states_from_folders(
                    results_dir,
                    folders,
                    file_name,
                    labels,
                    index_img,
                    index_txt,
                    prompt_len=prompt_len,
                    hidden_states_key=hidden_states_key,
                )

                suffix_ = suffix + f"epoch{e1}{e2}"

                for i, hskey in enumerate(hidden_states_key):

                    print(hskey)
                    if hskey != "hidden_states":
                        suffix_ = suffix + f"_{hskey}_"

                    if len(hidden_states_key) > 1:
                        all_embed_txt1_ = [embds[i] for embds in all_embed_txt1]
                        all_embed_img1_ = [embds[i] for embds in all_embed_img1]

                        all_embed_txt2_ = [embds[i] for embds in all_embed_txt2]
                        all_embed_img2_ = [embds[i] for embds in all_embed_img2]
                    else:
                        all_embed_txt1_ = all_embed_txt1
                        all_embed_img1_ = all_embed_img1

                        all_embed_txt2_ = all_embed_txt2
                        all_embed_img2_ = all_embed_img2

                    plot_layers = list(range(32))
                    all_avg, all_std = compute_sims_from_hs_across_layers(
                        all_embed_img1_,
                        all_embed_img2_,
                        new_labels,
                        plot_layers,
                        sim_mode=args.sim_mode,
                        textlab="P2",
                    )

                    avgs1.append(all_avg)
                    stds1.append(all_std)

                    plt_avg_std_sims(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_layers_avg_p1p2",
                        save_path=save_results_dir,
                        fontsize=30,
                        promptlab="P1",
                        textlab="P2",
                    )
                    plt_avg_std_sims(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_layers_avg_skip_pp_p1p2",
                        save_path=save_results_dir,
                        fontsize=30,
                        skip_pp=True,
                        promptlab="P1",
                        textlab="P2",
                    )
                    plt_avg_std_sims(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_layers_avg_plot_tt_p1p2",
                        save_path=save_results_dir,
                        fontsize=30,
                        plot_tt=True,
                        promptlab="P1",
                        textlab="P2",
                    )

                    plot_layers = list(range(32))
                    all_avg, all_std, labels_ = compute_sims_from_hs_across_layers(
                        all_embed_img1_,
                        all_embed_img2_,
                        new_labels,
                        plot_layers,
                        per_modality=True,
                        unnormalized_sim=False,
                        sim_mode=args.sim_mode,
                        textlab="P2",
                    )

                    plt_avg_std_sims_permodality(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_layers_avg_permodal_p1p2",
                        save_path=save_results_dir,
                        labels=labels_,
                    )
                    plt_avg_std_sims_permodality(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_layers_avg_permodal_skip_pp_p1p2",
                        save_path=save_results_dir,
                        labels=labels_,
                        skip_pp=True,
                    )
                    plt_avg_std_sims_permodality(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_layers_avg_permodal_plot_tt_p1p2",
                        save_path=save_results_dir,
                        labels=labels_,
                        plot_tt=True,
                    )

                    plot_layers = list(range(32))
                    all_avg, all_std = compute_sims_from_hs_across_layers(
                        all_embed_txt1_,
                        all_embed_txt2_,
                        new_labels,
                        plot_layers,
                        sim_mode=args.sim_mode,
                        textlab="T2",
                    )

                    avgs2.append(all_avg)
                    stds2.append(all_std)

                    plt_avg_std_sims(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_layers_avg_t1t2",
                        save_path=save_results_dir,
                        fontsize=30,
                        promptlab="T1",
                        textlab="T2",
                    )
                    plt_avg_std_sims(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_layers_avg_skip_pp_t1t2",
                        save_path=save_results_dir,
                        fontsize=30,
                        skip_pp=True,
                        promptlab="T1",
                        textlab="T2",
                    )
                    plt_avg_std_sims(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_layers_avg_plot_tt_t1t2",
                        save_path=save_results_dir,
                        fontsize=30,
                        plot_tt=True,
                        promptlab="T1",
                        textlab="T2",
                    )

                    plot_layers = list(range(32))
                    all_avg, all_std, labels_ = compute_sims_from_hs_across_layers(
                        all_embed_txt1_,
                        all_embed_txt2_,
                        new_labels,
                        plot_layers,
                        per_modality=True,
                        unnormalized_sim=False,
                        sim_mode=args.sim_mode,
                        textlab="T2",
                    )

                    plt_avg_std_sims_permodality(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_layers_avg_permodal_t1t2",
                        save_path=save_results_dir,
                        labels=labels_,
                    )
                    plt_avg_std_sims_permodality(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_layers_avg_permodal_skip_pp_t1t2",
                        save_path=save_results_dir,
                        labels=labels_,
                        skip_pp=True,
                    )
                    plt_avg_std_sims_permodality(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_layers_avg_permodal_plot_tt_t1t2",
                        save_path=save_results_dir,
                        labels=labels_,
                        plot_tt=True,
                    )

            epochs_ = ["ep 0-1", "ep 1-2", "ep 2-3"]

            plt_avg_std_sims_across(
                avgs1,
                stds1,
                plot_layers,
                plot_name=f"{suffix}emb_sim_layers_avg_all_epochs_p1p2",
                save_path=save_results_dir,
                labels=epochs_,
                promptlab="P1",
                textlab="P2",
            )
            plt_avg_std_sims_across(
                avgs1,
                stds1,
                plot_layers,
                plot_name=f"{suffix}emb_sim_layers_avg_all_epochs_skip_pp_p1p2",
                save_path=save_results_dir,
                labels=epochs_,
                skip_pp=True,
                promptlab="P1",
                textlab="P2",
            )
            plt_avg_std_sims_across(
                avgs1,
                stds1,
                plot_layers,
                plot_name=f"{suffix}emb_sim_layers_avg_all_epochs_plot_tt_p1p2",
                save_path=save_results_dir,
                labels=epochs_,
                plot_tt=True,
                promptlab="P1",
                textlab="P2",
            )

            plt_avg_std_sims_across(
                avgs2,
                stds2,
                plot_layers,
                plot_name=f"{suffix}emb_sim_layers_avg_all_epochs_t1t2",
                save_path=save_results_dir,
                labels=epochs_,
                promptlab="T1",
                textlab="T2",
            )
            plt_avg_std_sims_across(
                avgs2,
                stds2,
                plot_layers,
                plot_name=f"{suffix}emb_sim_layers_avg_all_epochs_skip_pp_t1t2",
                save_path=save_results_dir,
                labels=epochs_,
                skip_pp=True,
                promptlab="T1",
                textlab="T2",
            )
            plt_avg_std_sims_across(
                avgs2,
                stds2,
                plot_layers,
                plot_name=f"{suffix}emb_sim_layers_avg_all_epochs_plot_tt_t1t2",
                save_path=save_results_dir,
                labels=epochs_,
                plot_tt=True,
                promptlab="T1",
                textlab="T2",
            )

        elif "epochs" in args.mode:
            avgs = []
            stds = []

            if "allepochs" in args.mode:

                suffix += "_allepochs"
                epochs = [-1, 0, 1, 2, 3, 4]
            else:
                epochs = [-1, 0, 1, 2]

            if "opt" in args.model:
                epochs = [0, 3, 6, 12]

            if "skipfirst" in args.mode:

                suffix += "_skipfirst"
                epochs = epochs[1:]

            if "task" in args.mode:
                for i, folder_ in enumerate(folders):
                    folder_ = [folder_]
                    labels_ = [labels[i]]
                    for e in epochs:
                        print(f"epoch: {e}")

                        if e >= 0:
                            file_name = f"checkpoint_{e}all_hidden_states.pth"
                        else:
                            file_name = f"scratchall_hidden_states.pth"

                        plot_suffix = file_name.split(".")[0]
                        (
                            all_embed_txt,
                            all_embed_img,
                            new_labels,
                            all_embed_img_before,
                            all_embed_img_connector,
                        ) = read_hidden_states_from_folders(
                            results_dir,
                            folder_,
                            file_name,
                            labels_,
                            index_img,
                            index_txt,
                            prompt_len=prompt_len,
                            hidden_states_key=hidden_states_key,
                        )

                        suffix_ = suffix + f"epoch{e}" + f"{labels_[0]}"
                        for i, hskey in enumerate(hidden_states_key):

                            print(hskey)
                            if hskey != "hidden_states":
                                suffix_ = suffix + f"_{hskey}_"

                            if len(hidden_states_key) > 1:
                                all_embed_txt_ = [embds[i] for embds in all_embed_txt]
                                all_embed_img_ = [embds[i] for embds in all_embed_img]
                            else:
                                all_embed_txt_ = all_embed_txt
                                all_embed_img_ = all_embed_img

                            plot_layers = list(range(32))
                            all_avg, all_std = compute_sims_from_hs_across_layers(
                                all_embed_img_,
                                all_embed_txt_,
                                new_labels,
                                plot_layers,
                                sim_mode=args.sim_mode,
                                img_layer=img_layer,
                            )

                            avgs.append(all_avg)
                            stds.append(all_std)

                            plt_avg_std_sims(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{suffix_}emb_sim_layers_avg",
                                save_path=save_results_dir,
                                fontsize=30,
                            )
                            plt_avg_std_sims(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{suffix_}emb_sim_layers_avg_skip_pp",
                                save_path=save_results_dir,
                                fontsize=30,
                                skip_pp=True,
                            )
                            plt_avg_std_sims(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{suffix_}emb_sim_layers_avg_plot_tt",
                                save_path=save_results_dir,
                                fontsize=30,
                                plot_tt=True,
                            )

                    epochs_ = [f"ep {e}" for e in epochs]

                    plt_avg_std_sims_across(
                        avgs,
                        stds,
                        plot_layers,
                        plot_name=f"{suffix}emb_sim_layers_avg_all_epochs",
                        save_path=save_results_dir,
                        labels=epochs_,
                    )
                    plt_avg_std_sims_across(
                        avgs,
                        stds,
                        plot_layers,
                        plot_name=f"{suffix}emb_sim_layers_avg_all_epochs_skip_pp",
                        save_path=save_results_dir,
                        labels=epochs_,
                        skip_pp=True,
                    )
                    plt_avg_std_sims_across(
                        avgs,
                        stds,
                        plot_layers,
                        plot_name=f"{suffix}emb_sim_layers_avg_all_epochs_plot_tt",
                        save_path=save_results_dir,
                        labels=epochs_,
                        plot_tt=True,
                    )

            else:

                for e in epochs:
                    print(f"epoch: {e}")

                    if e >= 0:
                        file_name = f"checkpoint_{e}all_hidden_states.pth"
                    else:
                        file_name = f"scratchall_hidden_states.pth"

                    plot_suffix = file_name.split(".")[0]
                    (
                        all_embed_txt,
                        all_embed_img,
                        new_labels,
                        all_embed_img_before,
                        all_embed_img_connector,
                    ) = read_hidden_states_from_folders(
                        results_dir,
                        folders,
                        file_name,
                        labels,
                        index_img,
                        index_txt,
                        prompt_len=prompt_len,
                        hidden_states_key=hidden_states_key,
                    )

                    suffix_ = suffix + f"epoch{e}"
                    for i, hskey in enumerate(hidden_states_key):

                        print(hskey)
                        if hskey != "hidden_states":
                            suffix_ = suffix + f"_{hskey}_"

                        if len(hidden_states_key) > 1:
                            all_embed_txt_ = [embds[i] for embds in all_embed_txt]
                            all_embed_img_ = [embds[i] for embds in all_embed_img]
                        else:
                            all_embed_txt_ = all_embed_txt
                            all_embed_img_ = all_embed_img
                        plot_layers = list(range(32))

                        if "consecutive" in args.mode:
                            figsize = (25, 20)
                            font_offset = 15
                            legend_pos = "lower center"

                            ### norms
                            (
                                all_avg,
                                all_std,
                            ) = compute_norm_from_hs_across_consecutive_layers(
                                all_embed_img,
                                all_embed_txt,
                                plot_layers,
                                sim_mode=args.sim_mode,
                            )

                            plt_avg_std_consecutive_layers_sims(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{suffix_}emb_norm_consecutive_layers_avg",
                                save_path=save_results_dir,
                                y_label="Tokens Norm",
                            )

                            avgs.append(all_avg)
                            stds.append(all_std)

                            ### norms
                            (
                                all_avg,
                                all_std,
                            ) = compute_norm_from_hs_across_consecutive_layers(
                                all_embed_img,
                                all_embed_txt,
                                plot_layers,
                                per_modality=True,
                                sim_mode=args.sim_mode,
                            )

                            all_avg_, all_std_ = organise_per_modal_dict(
                                all_avg, all_std
                            )

                            plt_avg_std_consecutive_layers_sims(
                                all_avg_,
                                all_std_,
                                plot_layers,
                                plot_name=f"{suffix_}emb_norm_consecutive_layers_avg_permodal",
                                save_path=save_results_dir,
                                y_label="Tokens Norm",
                                labels=labels,
                                legend_pos=legend_pos,
                                figsize=figsize,
                            )

                        else:
                            all_avg, all_std = compute_sims_from_hs_across_layers(
                                all_embed_img_,
                                all_embed_txt_,
                                new_labels,
                                plot_layers,
                                sim_mode=args.sim_mode,
                                img_layer=img_layer,
                            )

                            avgs.append(all_avg)
                            stds.append(all_std)

                            plt_avg_std_sims(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{suffix_}emb_sim_layers_avg",
                                save_path=save_results_dir,
                                fontsize=30,
                            )
                            plt_avg_std_sims(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{suffix_}emb_sim_layers_avg_skip_pp",
                                save_path=save_results_dir,
                                fontsize=30,
                                skip_pp=True,
                            )
                            plt_avg_std_sims(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{suffix_}emb_sim_layers_avg_plot_tt",
                                save_path=save_results_dir,
                                fontsize=30,
                                plot_tt=True,
                            )

                            (
                                all_avg,
                                all_std,
                                labels_,
                            ) = compute_sims_from_hs_across_layers(
                                all_embed_img_,
                                all_embed_txt_,
                                new_labels,
                                plot_layers,
                                per_modality=True,
                                unnormalized_sim=False,
                                sim_mode=args.sim_mode,
                                img_layer=img_layer,
                            )
                            # labels_ = [labels, labels, labels]
                            plt_avg_std_sims_permodality(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{suffix_}emb_sim_layers_avg_permodal",
                                save_path=save_results_dir,
                                labels=labels_,
                            )
                            plt_avg_std_sims_permodality(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{suffix_}emb_sim_layers_avg_permodal_skip_pp",
                                save_path=save_results_dir,
                                labels=labels_,
                                skip_pp=True,
                            )
                            plt_avg_std_sims_permodality(
                                all_avg,
                                all_std,
                                plot_layers,
                                plot_name=f"{suffix_}emb_sim_layers_avg_permodal_plot_tt",
                                save_path=save_results_dir,
                                labels=labels_,
                                plot_tt=True,
                            )

                # epochs_ = ['ep 0', 'ep 1', 'ep 2', 'ep 3']

                epochs_ = [f"ep {e}" for e in epochs]
                if "consecutive" in args.mode:
                    plt_avg_std_consecutive_layers_sims_across(
                        avgs,
                        stds,
                        plot_layers,
                        plot_name=f"{suffix}emb_norm_layers_avg_all_epochs",
                        save_path=save_results_dir,
                        metalabels=epochs_,
                        labels=labels,
                    )

                else:
                    plt_avg_std_sims_across(
                        avgs,
                        stds,
                        plot_layers,
                        plot_name=f"{suffix}emb_sim_layers_avg_all_epochs",
                        save_path=save_results_dir,
                        labels=epochs_,
                    )
                    plt_avg_std_sims_across(
                        avgs,
                        stds,
                        plot_layers,
                        plot_name=f"{suffix}emb_sim_layers_avg_all_epochs_skip_pp",
                        save_path=save_results_dir,
                        labels=epochs_,
                        skip_pp=True,
                    )
                    plt_avg_std_sims_across(
                        avgs,
                        stds,
                        plot_layers,
                        plot_name=f"{suffix}emb_sim_layers_avg_all_epochs_plot_tt",
                        save_path=save_results_dir,
                        labels=epochs_,
                        plot_tt=True,
                    )

        else:

            plot_suffix = file_name.split(".")[0]
            (
                all_embed_txt,
                all_embed_img,
                new_labels,
                all_embed_img_before,
                all_embed_img_connector,
            ) = read_hidden_states_from_folders(
                results_dir,
                folders,
                file_name,
                labels,
                index_img,
                index_txt,
                prompt_len=prompt_len,
                hidden_states_key=hidden_states_key,
            )
            if "inside" in args.mode:
                plot_layers = list(range(1, 33))
            else:
                plot_layers = list(range(32))

            all_avg_sims = []
            all_std_sims = []

            all_avg_norm = []
            all_std_norm = []

            if "consecutive" not in args.mode:

                if "intra" in args.mode:
                    compute_intra = True
                else:
                    compute_intra = False

                suffix_ = suffix
                for i, hskey in enumerate(hidden_states_key):

                    print(hskey)
                    if hskey != "hidden_states":
                        suffix_ = suffix + f"_{hskey}_"

                    if len(hidden_states_key) > 1:
                        all_embed_txt_ = [embds[i] for embds in all_embed_txt]
                        all_embed_img_ = [embds[i] for embds in all_embed_img]
                    else:
                        all_embed_txt_ = all_embed_txt
                        all_embed_img_ = all_embed_img

                    all_avg, all_std = compute_sims_from_hs_across_layers(
                        all_embed_img_,
                        all_embed_txt_,
                        new_labels,
                        plot_layers,
                        sim_mode=args.sim_mode,
                        compute_intra=compute_intra,
                    )

                    all_avg_sims.append(all_avg)
                    all_std_sims.append(all_std)

                    plt_avg_std_sims(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_layers_avg",
                        save_path=save_results_dir,
                        fontsize=30,
                    )
                    plt_avg_std_sims(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_layers_avg_skip_pp",
                        save_path=save_results_dir,
                        fontsize=30,
                        skip_pp=True,
                    )
                    plt_avg_std_sims(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_layers_avg_plot_tt",
                        save_path=save_results_dir,
                        fontsize=30,
                        plot_tt=True,
                    )

                    if compute_intra:
                        plt_avg_std_sims(
                            all_avg,
                            all_std,
                            plot_layers,
                            plot_name=f"{suffix_}emb_sim_layers_avg_plot_tt_pp",
                            save_path=save_results_dir,
                            fontsize=30,
                            plot_tt=True,
                            plot_intra=True,
                        )

                    all_avg, all_std, labels_ = compute_sims_from_hs_across_layers(
                        all_embed_img_,
                        all_embed_txt_,
                        new_labels,
                        plot_layers,
                        per_modality=True,
                        unnormalized_sim=False,
                        sim_mode=args.sim_mode,
                        compute_intra=compute_intra,
                    )

                    plt_avg_std_sims_permodality(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_layers_avg_permodal",
                        save_path=save_results_dir,
                        labels=labels_,
                    )
                    plt_avg_std_sims_permodality(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_layers_avg_permodal_skip_pp",
                        save_path=save_results_dir,
                        labels=labels_,
                        skip_pp=True,
                    )
                    plt_avg_std_sims_permodality(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_layers_avg_permodal_plot_tt",
                        save_path=save_results_dir,
                        labels=labels_,
                        plot_tt=True,
                    )

                    if compute_intra:
                        plt_avg_std_sims_permodality(
                            all_avg,
                            all_std,
                            plot_layers,
                            plot_name=f"{suffix_}emb_sim_layers_avg_permodal_plot_tt_pp",
                            save_path=save_results_dir,
                            labels=labels_,
                            plot_intra=True,
                        )

                if "inside" in args.mode:

                    key_labels = [hskey_2_hsname[hskey] for hskey in hidden_states_key]
                    # key_labels = ['||LN1||', '||SA||', '||X + SA||', '||LN2||', '||FC1||', '||FC2||', '||X + FC2||']
                    ids_to_keep = [0, 1, 3, 4, 5]

                    key_labels_ = [key_labels[i] for i in ids_to_keep]
                    all_avg_sims_ = [all_avg_sims[i] for i in ids_to_keep]
                    all_std_sims_ = [all_std_sims[i] for i in ids_to_keep]

                    legend_pos = "lower right"

                    figsize = (30, 25)
                    font_offset = 15
                    plot_layers = list(range(32))
                    plt_avg_std_across_modalities_layers_sims(
                        all_avg_sims_,
                        all_std_sims_,
                        plot_layers,
                        plot_name=f"{suffix}emb_sim_all_inside_layers_avg",
                        save_path=save_results_dir,
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                    )

                    plt_avg_std_across_modalities_layers_sims(
                        all_avg_sims_,
                        all_std_sims_,
                        plot_layers,
                        plot_name=f"{suffix}emb_sim_all_inside_layers_avg_skip_pp",
                        save_path=save_results_dir,
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                        skip_pp=True,
                    )

                    plt_avg_std_across_modalities_layers_sims(
                        all_avg_sims_,
                        all_std_sims_,
                        plot_layers,
                        plot_name=f"{suffix}emb_diff_sim_all_inside_layers_avg",
                        save_path=save_results_dir,
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                        plot_difference=True,
                    )

                    ids_to_keep = [0, 1, 2, 3, 4, 5, 6]

                    key_labels_ = [key_labels[i] for i in ids_to_keep]
                    all_avg_sims_ = [all_avg_sims[i] for i in ids_to_keep]
                    all_std_sims_ = [all_std_sims[i] for i in ids_to_keep]

                    plt_avg_std_across_modalities_layers_sims(
                        all_avg_sims_,
                        all_std_sims_,
                        plot_layers,
                        plot_name=f"{suffix}emb_sim_all_inside_layers_avg_allin",
                        save_path=save_results_dir,
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                    )

                    plt_avg_std_across_modalities_layers_sims(
                        all_avg_sims_,
                        all_std_sims_,
                        plot_layers,
                        plot_name=f"{suffix}emb_sim_all_inside_layers_avg_skip_pp_allin",
                        save_path=save_results_dir,
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                        skip_pp=True,
                    )

                    plt_avg_std_across_modalities_layers_sims(
                        all_avg_sims_,
                        all_std_sims_,
                        plot_layers,
                        plot_name=f"{suffix}emb_diff_sim_all_inside_layers_avg_allin",
                        save_path=save_results_dir,
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                        plot_difference=True,
                    )

            else:

                if "inside" in args.mode:
                    intervals = [1]

                suffix_ = suffix
                for i, hskey in enumerate(hidden_states_key):

                    print(hskey)
                    if hskey != "hidden_states" or (
                        hskey == "hidden_states" and "inside" in args.mode
                    ):
                        suffix_ = suffix + f"_{hskey}_"

                    if len(hidden_states_key) > 1:
                        all_embed_txt_ = [embds[i] for embds in all_embed_txt]
                        all_embed_img_ = [embds[i] for embds in all_embed_img]
                    else:
                        all_embed_txt_ = all_embed_txt
                        all_embed_img_ = all_embed_img

                    all_avg, all_std = compute_sims_from_hs_across_consecutive_layers(
                        all_embed_img_,
                        all_embed_txt_,
                        plot_layers,
                        intervals,
                        sim_mode=args.sim_mode,
                    )

                    plt_avg_std_consecutive_layers_sims(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_consecutive_layers_avg",
                        save_path=save_results_dir,
                    )

                    all_avg_sims.append(all_avg)
                    all_std_sims.append(all_std)

                    ### norms
                    all_avg, all_std = compute_norm_from_hs_across_consecutive_layers(
                        all_embed_img_,
                        all_embed_txt_,
                        plot_layers,
                        sim_mode=args.sim_mode,
                    )

                    plt_avg_std_consecutive_layers_sims(
                        all_avg,
                        all_std,
                        plot_layers,
                        plot_name=f"{suffix_}emb_norm_consecutive_layers_avg",
                        save_path=save_results_dir,
                        y_label="Tokens Norm",
                    )

                    all_avg_norm.append(all_avg)
                    all_std_norm.append(all_std)

                    ## per modal
                    all_avg, all_std = compute_sims_from_hs_across_consecutive_layers(
                        all_embed_img_,
                        all_embed_txt_,
                        plot_layers,
                        intervals,
                        per_modality=True,
                        sim_mode=args.sim_mode,
                    )

                    all_avg_, all_std_ = organise_per_modal_dict(all_avg, all_std)

                    figsize = (25, 20)
                    font_offset = 15
                    legend_pos = "lower center"

                    plt_avg_std_consecutive_layers_sims(
                        all_avg_,
                        all_std_,
                        plot_layers,
                        plot_name=f"{suffix_}emb_sim_consecutive_layers_avg_permodal",
                        save_path=save_results_dir,
                        labels=labels,
                        legend_pos=legend_pos,
                        figsize=figsize,
                    )

                    ### norms
                    all_avg, all_std = compute_norm_from_hs_across_consecutive_layers(
                        all_embed_img_,
                        all_embed_txt_,
                        plot_layers,
                        per_modality=True,
                        sim_mode=args.sim_mode,
                    )

                    all_avg_, all_std_ = organise_per_modal_dict(all_avg, all_std)

                    plt_avg_std_consecutive_layers_sims(
                        all_avg_,
                        all_std_,
                        plot_layers,
                        plot_name=f"{suffix_}emb_norm_consecutive_layers_avg_permodal",
                        save_path=save_results_dir,
                        y_label="Tokens Norm",
                        labels=labels,
                        legend_pos=legend_pos,
                        figsize=figsize,
                    )

                if "inside" in args.mode:
                    key_labels = [hskey_2_hsname[hskey] for hskey in hidden_states_key]

                    ids_to_keep = [0, 1, 3, 4, 5]

                    key_labels_ = [key_labels[i] for i in ids_to_keep]
                    all_avg_sims_ = [all_avg_sims[i] for i in ids_to_keep]
                    all_std_sims_ = [all_std_sims[i] for i in ids_to_keep]

                    all_avg_norm_ = [all_avg_norm[i] for i in ids_to_keep]
                    all_std_norm_ = [all_std_norm[i] for i in ids_to_keep]

                    legend_pos = "lower right"

                    figsize = (30, 25)
                    font_offset = 15
                    plot_layers = list(range(1, 33))
                    plt_avg_std_consecutive_layers_sims(
                        all_avg_sims_,
                        all_std_sims_,
                        plot_layers,
                        plot_name=f"{suffix}emb_sim_all_inside_consecutive_layers_avg",
                        save_path=save_results_dir,
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                    )

                    plot_layers = list(range(1, 33))
                    plt_avg_std_consecutive_layers_sims(
                        all_avg_norm_,
                        all_std_norm_,
                        plot_layers,
                        plot_name=f"{suffix}emb_norm_all_inside_consecutive_layers_avg",
                        save_path=save_results_dir,
                        y_label="Tokens Norm",
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                    )

                    ids_to_keep = [0, 1, 2, 3, 4, 5, 6]
                    key_labels_ = [key_labels[i] for i in ids_to_keep]
                    all_avg_sims_ = [all_avg_sims[i] for i in ids_to_keep]
                    all_std_sims_ = [all_std_sims[i] for i in ids_to_keep]

                    all_avg_norm_ = [all_avg_norm[i] for i in ids_to_keep]
                    all_std_norm_ = [all_std_norm[i] for i in ids_to_keep]

                    legend_pos = "lower right"

                    figsize = (30, 25)
                    font_offset = 15
                    plot_layers = list(range(1, 33))
                    plt_avg_std_consecutive_layers_sims(
                        all_avg_sims_,
                        all_std_sims_,
                        plot_layers,
                        plot_name=f"{suffix}emb_sim_all_inside_consecutive_layers_avg_allin",
                        save_path=save_results_dir,
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                    )

                    plot_layers = list(range(1, 33))
                    plt_avg_std_consecutive_layers_sims(
                        all_avg_norm_,
                        all_std_norm_,
                        plot_layers,
                        plot_name=f"{suffix}emb_norm_all_inside_consecutive_layers_avg_allin",
                        save_path=save_results_dir,
                        y_label="Tokens Norm",
                        labels=key_labels_,
                        figsize=figsize,
                        font_offset=font_offset,
                        legend_pos=legend_pos,
                    )
