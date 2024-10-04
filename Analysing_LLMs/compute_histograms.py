import argparse
import os
import sys

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import LlamaForCausalLM, OPTForCausalLM
from utils import (read_hidden_states_from_folders,
                   read_hidden_states_from_folders_llava)

sys.path.append("~/ima/Analysing_LLMs")


def word2vec_score(w1s, w2s, nlp=None, mode="max"):
    scores = []
    print(f"Computing word2vec sim ...")
    for (
        w1,
        w2,
    ) in tqdm(zip(w1s, w2s)):
        t1s = nlp(w1)
        t2s = nlp(w2)

        sims = []
        for t1 in t1s:
            for t2 in t2s:
                sims.append(t1.similarity(t2))

        if sims:
            sims = np.array(sims)

            if mode == "max":
                score = np.max(sims)
            else:
                score = np.mean(sims)
            scores.append(score)

    return np.array(scores)


def kl_distance(p, q):
    # Compute KL divergence for each element
    kl_values = p * torch.log(p / q)

    # Handle cases where p or q is zero to avoid NaN values
    kl_values[(p == 0) | (q == 0)] = 0

    # Sum the KL divergence values
    kl_div = torch.sum(kl_values)

    return kl_div


def filter_english_words(text, english_words, remove_words=None):
    word_list = text  # .split(' ')
    word_list = [word.strip().lower() for word in word_list]
    word_list = [word for word in word_list if word.strip().lower() in english_words]
    word_list = [word for word in word_list if len(word) > 2]
    if remove_words is not None:
        word_list = [word for word in word_list if word not in remove_words]

    word_list = list(set(word_list))
    return " ".join(word_list)


def get_text_pred_from_ids(ids, tokenizer, english_words=None, remove_words=None):
    index = ids  # .squeeze(-1)
    text = tokenizer.batch_decode(index, skip_special_tokens=True)  # [0]

    text = filter_english_words(text, english_words, remove_words=remove_words)
    return text


def get_text_pred(token, k, token_id, lm_head, tokenizer):
    with torch.no_grad():
        logits = lm_head(token)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        index = probs.topk(dim=-1, k=k)[-1]
        if token_id > 0:
            index = index[:, token_id, :]
        else:
            index = index.squeeze(-1)
        text = tokenizer.batch_decode(index, skip_special_tokens=True)[0]

        text = text.replace("\n", "")
    return text


def get_proba(token, lm_head, k=10, layer_norm=None):
    with torch.no_grad():

        token = torch.tensor(token)
        if layer_norm is not None:
            token = layer_norm(token)

        logits = lm_head(token)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        index = probs.topk(dim=-1, k=k)[-1]
    return probs, index


def compute_hist_from_hs_across_layers(
    all_embed_img,
    all_embed_txt,
    new_labels,
    layers=[0, 32],
    per_modality=False,
    lm_head=False,
    img_layer=None,
    k=10,
    get_proba_dist=False,
    layer_norm=None,
    mode="logit",
    flatten=True,
):

    labels = []
    all_ids = []

    for layer in tqdm(layers):
        img_txt_sims = []

        if img_layer is None:
            img_layer = layer

        for i in tqdm(range(len(all_embed_img))):

            img = [im[img_layer] for im in all_embed_img[i]]
            txt = [im[layer] for im in all_embed_txt[i]]

            labl = f"{new_labels[i]} vs T"

            img_txt_sims.append(
                compute_top_token_ids(
                    img,
                    txt,
                    lm_head=lm_head,
                    k=k,
                    get_probas=get_proba_dist,
                    layer_norm=layer_norm,
                    flatten=flatten,
                )
                + (labl,)
            )

        if per_modality:

            all_img_ids = [a[0] for a in img_txt_sims]
            all_txt_ids = [a[1] for a in img_txt_sims]
            labs = [a[2] for a in img_txt_sims]

        else:
            all_img_ids = []
            all_txt_ids = []
            for a in img_txt_sims:
                all_img_ids += a[0]  # len = 200 examples
                all_txt_ids += a[1]

            print(len(all_img_ids), len(all_txt_ids))
            if flatten:
                all_img_ids = [all_img_ids]
                all_txt_ids = [all_txt_ids]

                print(len(all_img_ids), len(all_txt_ids))

        if per_modality:
            labels.append(labs)
        all_ids.append([all_img_ids, all_txt_ids])

    if per_modality:
        return all_ids, labels
    else:
        return all_ids


def plot_histograms(
    all_img_ids,
    all_txt_ids,
    fontsize=30,
    font_offset=15,
    markersize=16,
    save_path=None,
    facecolor="lavender",
    plot_name="",
    labels=[""],
    bins=100,
    max_xrange=-1,
    max_yrange=-1,
    kde_plot=False,
):

    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams["lines.linewidth"] = 4.0
    plt.rcParams["font.family"] = "serif"  # Times New Roman serif
    plt.figure(figsize=(20, 15))
    ax = plt.axes()
    ax.set_facecolor(facecolor)

    for img_ids, txt_ids, label in tqdm(zip(all_img_ids, all_txt_ids, labels)):

        img_ids = [float(im[0]) for im in img_ids]
        txt_ids = [float(im[0]) for im in txt_ids]

        print(len(img_ids), label, img_ids[:10], txt_ids[:10], f"{label}-P")
        if kde_plot:
            sns.kdeplot(img_ids, label=f"{label}-P", alpha=0.3, fill=True)
            sns.kdeplot(txt_ids, label=f"{label}-T", alpha=0.3, fill=True)
        else:
            plt.hist(
                img_ids,
                bins=bins,
                align="left",
                rwidth=0.8,
                label=f"{label} (P)",
                alpha=0.6,
                histtype="stepfilled",
            )  # range(min(img_ids), max(img_ids) + 1)
            plt.hist(
                txt_ids,
                bins=bins,
                align="left",
                rwidth=0.8,
                label=f"{label} (T)",
                alpha=0.6,
                histtype="stepfilled",
            )

    if max_xrange > 0:
        plt.xlim(0, max_xrange)
    if max_yrange > 0:
        plt.ylim(0, max_yrange)

    plt.ylabel("Frequency", fontsize=fontsize + font_offset)
    plt.xlabel("Vocab", fontsize=fontsize + font_offset)
    plt.legend(fontsize=fontsize + font_offset)

    plt.grid(color="white", linewidth=4)

    plt.xticks(fontsize=fontsize + font_offset)
    plt.yticks(fontsize=fontsize + font_offset)

    if save_path is not None:
        save_plot_path = os.path.join(save_path, f"{plot_name}.jpg")
        plt.savefig(save_plot_path, bbox_inches="tight")
        print(f"saved at {save_plot_path}")


def compute_top_token_ids(
    list_1,
    list_2,
    lm_head=None,
    k=10,
    get_probas=False,
    num_examples=-1,
    layer_norm=None,
    flatten=True,
):

    all_ids1 = []
    all_ids2 = []
    all_ps1 = []
    all_ps2 = []

    if num_examples > 0:
        list_1, list_2 = list_1[:num_examples], list_2[:num_examples]

    all_ids1, all_ids2 = [], []
    for i, (l1, l2) in enumerate(zip(list_1, list_2)):
        # l1, l2 = l1.astype(float),  l2.astype(float)
        if l1.ndim < 2:
            l1, l2 = l1.unsqueeze(0), l2.unsqueeze(0)
        if lm_head.weight.dtype == torch.float16:
            l1, l2 = torch.tensor(l1).half().numpy(), torch.tensor(l2).half().numpy()
        else:
            l1, l2 = l1.astype(np.single), l2.astype(np.single)

        p1, id1 = get_proba(l1, lm_head, k=k, layer_norm=layer_norm)
        p2, id2 = get_proba(l2, lm_head, k=k, layer_norm=layer_norm)

        if flatten:
            if i == 0:
                all_ps1 = p1
                all_ps2 = p2

                all_ids1 = id1
                all_ids2 = id2
            else:
                all_ps1 = torch.cat((all_ps1, p1), dim=0)
                all_ps2 = torch.cat((all_ps2, p2), dim=0)

                all_ids1 = torch.cat((all_ids1, id1), dim=0)
                all_ids2 = torch.cat((all_ids2, id2), dim=0)
        else:
            all_ids1.append(id1)
            all_ids2.append(id2)

    if get_probas:
        distance = [kl_distance(l1, l2) for l1, l2 in zip(all_ps1, all_ps2)]
        return distance
    else:
        return all_ids1, all_ids2


def plt_avg_std_probadistance(
    avg,
    std,
    xaxis,
    fontsize=30,
    font_offset=15,
    markersize=16,
    save_path=None,
    facecolor="lavender",
    plot_name="",
    labels=[""],
    skip_pp=False,
    per_modality=False,
    solo=False,
    ylabel="Distance",
):

    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams["lines.linewidth"] = 4.0
    plt.rcParams["font.family"] = "serif"  # Times New Roman serif
    plt.figure(figsize=(20, 15))
    ax = plt.axes()
    ax.set_facecolor(facecolor)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    color_idx = 0

    if solo:
        p_lab = "-P"
        t_lab = "-T"
    else:
        p_lab = "-P vs P"
        t_lab = "-P vs T"

    if per_modality:

        for j in range(len(avg[0][0])):

            lab = labels[j]
            averages = np.array([a[0][j] for a in avg])
            std_devs = np.array([a[0][j] for a in std])

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
                    label=f"{lab}{p_lab}",
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

            averages = np.array([a[1][j] for a in avg])
            std_devs = np.array([a[1][j] for a in std])

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
                label=f"{lab}{t_lab}",
                color=color,
            )
            plt.fill_between(
                xaxis, averages - std_devs, averages + std_devs, alpha=0.15, color=color
            )
            plt.axhline(y=np.mean(averages), linestyle="--", color=color, zorder=3)

    else:

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
                label=f"{p_lab}",
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
            label=f"{t_lab}",
            color=color,
        )
        plt.fill_between(
            xaxis, averages - std_devs, averages + std_devs, alpha=0.15, color=color
        )
        plt.axhline(y=np.mean(averages), linestyle="--", color=color, zorder=3)

    plt.ylabel(ylabel, fontsize=fontsize + font_offset)
    plt.xlabel("Layers", fontsize=fontsize + font_offset)
    plt.legend(fontsize=fontsize + font_offset)

    plt.grid(color="white", linewidth=4)

    plt.xticks(fontsize=fontsize + font_offset)
    plt.yticks(fontsize=fontsize + font_offset)

    if save_path is not None:
        save_plot_path = os.path.join(save_path, f"{plot_name}.jpg")
        plt.savefig(save_plot_path, bbox_inches="tight")
        print(f"saved at {save_plot_path}")


def compute_proba_distance_from_hs_across_layers(
    all_embed_img,
    all_embed_txt,
    new_labels,
    layers=[0, 32],
    lm_head=False,
    img_layer=None,
    k=10,
    get_proba_dist=True,
    skip_pp=True,
    num_examples=-1,
    per_modality=False,
    layer_norm=None,
    mode="logit",
):
    img_txt_sims = []

    all_avg = []
    all_std = []
    labels = []

    for layer in tqdm(layers):
        img_txt_sims = []

        if img_layer is None:
            img_layer = layer

        for i in range(len(all_embed_img)):

            img = [im[img_layer] for im in all_embed_img[i]]
            txt = [im[layer] for im in all_embed_txt[i]]

            labl = f"{new_labels[i]} vs T"

            img_txt_sims.append(
                compute_top_token_ids(
                    img,
                    txt,
                    lm_head=lm_head,
                    k=k,
                    get_probas=get_proba_dist,
                    num_examples=num_examples,
                    layer_norm=layer_norm,
                )
                + [
                    labl,
                ]
            )

        img_img_sims = []
        if not skip_pp:
            for i in range(len(all_embed_img)):
                for j in range(len(all_embed_img)):
                    if i < j:
                        img = [im[img_layer] for im in all_embed_img[i]]
                        txt = [im[layer] for im in all_embed_img[j]]

                        labl = f"{new_labels[i]} vs {new_labels[j]}"
                        img_img_sims.append(
                            compute_top_token_ids(
                                img,
                                txt,
                                lm_head=lm_head,
                                k=k,
                                get_probas=get_proba_dist,
                                num_examples=num_examples,
                                layer_norm=layer_norm,
                            )
                            + [
                                labl,
                            ]
                        )

        if per_modality:
            all_img_img = []
            all_img_txt = []
            labels_img = []
            labels_txt = []
            for i in range(len(img_txt_sims)):
                all_img_txt.append(np.array(img_txt_sims[i][:-1]))
                labels_txt.append(img_txt_sims[i][-1])

            if not skip_pp:
                for i in range(len(img_img_sims)):
                    all_img_img.append(np.array(img_img_sims[i][:-1]))
                    labels_img.append(img_img_sims[i][-1])

                all_avg.append(
                    [
                        [np.mean(s) for s in all_img_img],
                        [np.mean(s) for s in all_img_txt],
                    ]
                )
                all_std.append(
                    [[np.std(s) for s in all_img_img], [np.std(s) for s in all_img_txt]]
                )
                labels = [labels_img, labels_txt]
            else:

                all_avg.append([[], [s.mean() for s in all_img_txt]])
                all_std.append([[], [s.std() for s in all_img_txt]])
                labels = [labels_img, labels_txt]

        else:

            all_img_txt = np.concatenate(
                [np.array(s[:-1]) for s in img_txt_sims], axis=0
            )

            if not skip_pp:
                all_img_img = np.concatenate(
                    [np.array(s[:-1]) for s in img_img_sims], axis=0
                )

                all_avg.append([np.mean(all_img_img), np.mean(all_img_txt)])
                all_std.append([np.std(all_img_img), np.std(all_img_txt)])
            else:
                all_avg.append([0, np.mean(all_img_txt)])
                all_std.append([0, np.std(all_img_txt)])

    if per_modality:
        return all_avg, all_std, labels
    else:
        return all_avg, all_std


def entropy(distribution):
    neg_log_probs = -distribution * torch.log(distribution)
    entropy_value = torch.sum(neg_log_probs, dim=-1)

    return entropy_value


def compute_entropy(list_1, lm_head=None, k=10, num_examples=-1, layer_norm=None):

    all_ids1 = []
    all_ids2 = []
    all_ps1 = []
    all_ps2 = []

    if num_examples > 0:
        list_1 = list_1[:num_examples]

    for i, l1 in enumerate(list_1):

        if l1.ndim < 2:
            l1 = l1.unsqueeze(0)
        if lm_head.weight.dtype == torch.float16:
            l1 = torch.tensor(l1).half().numpy()
        else:
            l1 = l1.astype(np.single)

        p1, id1 = get_proba(l1, lm_head, k=k, layer_norm=layer_norm)

        if i == 0:

            all_ps1 = p1

            all_ids1 = id1

        else:

            all_ps1 = torch.cat((all_ps1, p1), dim=0)

            all_ids1 = torch.cat((all_ids1, id1), dim=0)

    distance = [entropy(l1) for l1 in all_ps1]

    return distance


def compute_proba_entropy_from_hs_across_layers(
    all_embed_img,
    all_embed_txt,
    new_labels,
    layers=[0, 32],
    lm_head=False,
    img_layer=None,
    k=10,
    get_proba_dist=True,
    skip_pp=True,
    num_examples=-1,
    per_modality=False,
    layer_norm=None,
    mode="logit",
):
    img_txt_sims = []

    all_avg = []
    all_std = []
    labels = []

    for layer in tqdm(layers):
        img_txt_sims = []
        img_img_sims = []

        if img_layer is None:
            img_layer = layer

        for i in range(len(all_embed_img)):

            img = [im[img_layer] for im in all_embed_img[i]]
            labl = f"{new_labels[i]}"
            img_img_sims.append(
                compute_entropy(
                    img,
                    lm_head=lm_head,
                    k=k,
                    num_examples=num_examples,
                    layer_norm=layer_norm,
                )
                + [
                    labl,
                ]
            )

            txt = [im[layer] for im in all_embed_txt[i]]
            labl = f"{new_labels[i]}"
            img_txt_sims.append(
                compute_entropy(
                    txt,
                    lm_head=lm_head,
                    k=k,
                    num_examples=num_examples,
                    layer_norm=layer_norm,
                )
                + [
                    labl,
                ]
            )

        if per_modality:
            all_img_img = []
            all_img_txt = []
            labels_img = []
            labels_txt = []
            for i in range(len(img_txt_sims)):
                all_img_txt.append(np.array(img_txt_sims[i][:-1]))
                labels_txt.append(img_txt_sims[i][-1])

            if not skip_pp:
                for i in range(len(img_img_sims)):
                    all_img_img.append(np.array(img_img_sims[i][:-1]))
                    labels_img.append(img_img_sims[i][-1])

                all_avg.append(
                    [
                        [np.mean(s) for s in all_img_img],
                        [np.mean(s) for s in all_img_txt],
                    ]
                )
                all_std.append(
                    [[np.std(s) for s in all_img_img], [np.std(s) for s in all_img_txt]]
                )
                labels = [labels_img, labels_txt]
            else:

                all_avg.append([[], [s.mean() for s in all_img_txt]])
                all_std.append([[], [s.std() for s in all_img_txt]])
                labels = [labels_img, labels_txt]

        else:

            all_img_txt = np.concatenate(
                [np.array(s[:-1]) for s in img_txt_sims], axis=0
            )

            if not skip_pp:
                all_img_img = np.concatenate(
                    [np.array(s[:-1]) for s in img_img_sims], axis=0
                )

                all_avg.append([np.mean(all_img_img), np.mean(all_img_txt)])
                all_std.append([np.std(all_img_img), np.std(all_img_txt)])
            else:
                all_avg.append([0, np.mean(all_img_txt)])
                all_std.append([0, np.std(all_img_txt)])

    if per_modality:
        return all_avg, all_std, labels
    else:
        return all_avg, all_std


def plt_avg_std_probadistance_llava(
    avg,
    std,
    xaxis,
    fontsize=30,
    font_offset=15,
    markersize=16,
    save_path=None,
    facecolor="lavender",
    plot_name="",
    labels=[""],
    skip_pp=False,
    per_modality=False,
    solo=False,
    ylabel="Distance",
):

    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams["lines.linewidth"] = 4.0
    plt.rcParams["font.family"] = "serif"  # Times New Roman serif
    plt.figure(figsize=(20, 15))
    ax = plt.axes()
    ax.set_facecolor(facecolor)

    if solo:
        p_lab = "P"
        t_lab = "T"
    else:
        p_lab = "P vs P"
        t_lab = "P vs T"

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    color_idx = 0

    if per_modality:

        for j in range(len(avg[0][0])):

            lab = labels[0][j]
            averages = np.array([a[0][j] for a in avg])
            std_devs = np.array([a[0][j] for a in std])

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
                    label=f"{lab}-{p_lab}",
                    color=color,
                )
                plt.fill_between(
                    xaxis,
                    averages - std_devs,
                    averages + std_devs,
                    alpha=0.2,
                    color=color,
                )
                plt.axhline(y=np.mean(averages), linestyle="--", color=color)

        for j in range(len(avg[0][1])):

            color = colors[color_idx]
            color_idx += 1
            if color_idx >= len(colors):
                color_idx = 0
            averages = np.array([a[1][j] for a in avg])
            std_devs = np.array([a[1][j] for a in std])

            lab = labels[1][j]
            plt.plot(
                xaxis,
                averages,
                marker="^",
                linestyle="--",
                markersize=markersize,
                label=f"{lab}-{t_lab}",
                color=color,
            )
            plt.fill_between(
                xaxis, averages - std_devs, averages + std_devs, alpha=0.15, color=color
            )
            plt.axhline(y=np.mean(averages), linestyle="--", color=color)

    else:

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
                label=f"{p_lab}",
                color=color,
            )
            plt.fill_between(
                xaxis, averages - std_devs, averages + std_devs, alpha=0.2, color=color
            )
            plt.axhline(y=np.mean(averages), linestyle="--", color=color)

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
            label=f"{t_lab}",
            color=color,
        )
        plt.fill_between(
            xaxis, averages - std_devs, averages + std_devs, alpha=0.15, color=color
        )
        plt.axhline(y=np.mean(averages), linestyle="--", color=color)

    plt.ylabel(ylabel, fontsize=fontsize + font_offset)
    plt.xlabel("Layers", fontsize=fontsize + font_offset)
    plt.legend(fontsize=fontsize + font_offset)

    plt.grid(color="white", linewidth=4)

    plt.xticks(fontsize=fontsize + font_offset)
    plt.yticks(fontsize=fontsize + font_offset)

    if save_path is not None:
        save_plot_path = os.path.join(save_path, f"{plot_name}.jpg")
        plt.savefig(save_plot_path, bbox_inches="tight")
        print(f"saved at {save_plot_path}")

    # plt.show()


def compute_proba_distance_from_hs_consecutive_across_layers(
    all_embed_img,
    all_embed_txt,
    new_labels,
    layers=[0, 32],
    lm_head=False,
    img_layer=None,
    k=10,
    get_proba_dist=True,
    skip_pp=True,
    num_examples=-1,
    per_modality=False,
    layer_norm=None,
    intervals=[1],
    mode="logit",
):
    img_txt_sims = []

    labels = []

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

            if img_layer is None:
                img_layer = layer

            for i in range(len(all_embed_img)):

                img = [im[img_layer] for im in all_embed_img[i]]
                txt = [im[layer + interval] for im in all_embed_img[i]]

                labl = f"{new_labels[i]}-P"
                img_img_sims.append(
                    compute_top_token_ids(
                        img,
                        txt,
                        lm_head=lm_head,
                        k=k,
                        get_probas=get_proba_dist,
                        num_examples=num_examples,
                        layer_norm=layer_norm,
                    )
                    + [
                        labl,
                    ]
                )

                img = [im[img_layer] for im in all_embed_txt[i]]
                txt = [im[layer + interval] for im in all_embed_txt[i]]

                labl = f"{new_labels[i]}-T"
                img_txt_sims.append(
                    compute_top_token_ids(
                        img,
                        txt,
                        lm_head=lm_head,
                        k=k,
                        get_probas=get_proba_dist,
                        num_examples=num_examples,
                        layer_norm=layer_norm,
                    )
                    + [
                        labl,
                    ]
                )

            if per_modality:
                all_img_img = []
                all_img_txt = []
                labels_img = []
                labels_txt = []
                for i in range(len(img_txt_sims)):
                    all_img_txt.append(np.array(img_txt_sims[i][:-1]))
                    labels_txt.append(img_txt_sims[i][-1])

                if not skip_pp:
                    for i in range(len(img_img_sims)):
                        all_img_img.append(np.array(img_img_sims[i][:-1]))
                        labels_img.append(img_img_sims[i][-1])

                    all_avg.append(
                        [
                            [np.mean(s) for s in all_img_img],
                            [np.mean(s) for s in all_img_txt],
                        ]
                    )
                    all_std.append(
                        [
                            [np.std(s) for s in all_img_img],
                            [np.std(s) for s in all_img_txt],
                        ]
                    )
                    labels = [labels_img, labels_txt]
                else:

                    all_avg.append([[], [s.mean() for s in all_img_txt]])
                    all_std.append([[], [s.std() for s in all_img_txt]])
                    labels = [labels_img, labels_txt]

            else:

                all_img_txt = np.concatenate(
                    [np.array(s[:-1]) for s in img_txt_sims], axis=0
                )

                if not skip_pp:
                    all_img_img = np.concatenate(
                        [np.array(s[:-1]) for s in img_img_sims], axis=0
                    )

                    all_avg.append([np.mean(all_img_img), np.mean(all_img_txt)])
                    all_std.append([np.std(all_img_img), np.std(all_img_txt)])
                else:
                    all_avg.append([0, np.mean(all_img_txt)])
                    all_std.append([0, np.std(all_img_txt)])

        avgs[interval] = all_avg
        stds[interval] = all_std

    if per_modality:
        return avgs, stds, labels
    else:
        return avgs, stds


def plt_avg_std_probadistance_consecutives(
    avgs,
    stds,
    xaxis,
    fontsize=30,
    font_offset=15,
    markersize=16,
    save_path=None,
    facecolor="lavender",
    plot_name="",
    labels=[""],
    skip_pp=False,
    per_modality=False,
    intervals=[1],
    keys_to_remove=[],
):

    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams["lines.linewidth"] = 4.0
    plt.rcParams["font.family"] = "serif"  # Times New Roman serif
    plt.figure(figsize=(20, 15))
    ax = plt.axes()
    ax.set_facecolor(facecolor)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    color_idx = 0

    for i, key in enumerate(avgs.keys()):

        if key in keys_to_remove:
            continue

        avg = avgs[key]
        std = stds[key]

        n = key
        if per_modality:

            for j in range(len(avg[0][0])):

                lab = labels[j]
                averages = np.array([a[0][j] for a in avg])
                std_devs = np.array([a[0][j] for a in std])

                xaxis_ = xaxis[: len(averages)]
                if not skip_pp:
                    color = colors[color_idx]
                    color_idx += 1
                    if color_idx >= len(colors):
                        color_idx = 0
                    plt.plot(
                        xaxis_,
                        averages,
                        marker="o",
                        linestyle="--",
                        markersize=markersize,
                        label=f"{lab} (n {n})",
                        color=color,
                    )
                    plt.fill_between(
                        xaxis_,
                        averages - std_devs,
                        averages + std_devs,
                        alpha=0.2,
                        color=color,
                    )
                    plt.axhline(y=np.mean(averages), linestyle="--", color=color)

                averages = np.array([a[1][j] for a in avg])
                std_devs = np.array([a[1][j] for a in std])

                color = colors[color_idx]
                color_idx += 1
                if color_idx >= len(colors):
                    color_idx = 0
                plt.plot(
                    xaxis_,
                    averages,
                    marker="^",
                    linestyle="--",
                    markersize=markersize,
                    label=f"{lab} (n {n})",
                    color=color,
                )
                plt.fill_between(
                    xaxis_,
                    averages - std_devs,
                    averages + std_devs,
                    alpha=0.15,
                    color=color,
                )
                plt.axhline(y=np.mean(averages), linestyle="--", color=color)

        else:

            averages = np.array([a[0] for a in avg])
            std_devs = np.array([a[0] for a in std])

            xaxis_ = xaxis[: len(averages)]

            if not skip_pp:
                color = colors[color_idx]
                color_idx += 1
                if color_idx >= len(colors):
                    color_idx = 0
                plt.plot(
                    xaxis_,
                    averages,
                    marker="o",
                    linestyle="--",
                    markersize=markersize,
                    label=f"P (n {n})",
                    color=color,
                )
                plt.fill_between(
                    xaxis_,
                    averages - std_devs,
                    averages + std_devs,
                    alpha=0.2,
                    color=color,
                )
                plt.axhline(y=np.mean(averages), linestyle="--", color=color)

            averages = np.array([a[1] for a in avg])
            std_devs = np.array([a[1] for a in std])

            color = colors[color_idx]
            color_idx += 1
            if color_idx >= len(colors):
                color_idx = 0
            plt.plot(
                xaxis_,
                averages,
                marker="^",
                linestyle="--",
                markersize=markersize,
                label=f"T (n {n})",
                color=color,
            )
            plt.fill_between(
                xaxis_,
                averages - std_devs,
                averages + std_devs,
                alpha=0.15,
                color=color,
            )
            plt.axhline(y=np.mean(averages), linestyle="--", color=color)

    plt.ylabel("Distance", fontsize=fontsize + font_offset)
    plt.xlabel("Layers", fontsize=fontsize + font_offset)
    plt.legend(fontsize=fontsize + font_offset)

    plt.grid(color="white", linewidth=4)

    plt.xticks(fontsize=fontsize + font_offset)
    plt.yticks(fontsize=fontsize + font_offset)

    if save_path is not None:
        save_plot_path = os.path.join(save_path, f"{plot_name}.jpg")
        plt.savefig(save_plot_path, bbox_inches="tight")
        print(f"saved at {save_plot_path}")

    # plt.show()


def plt_avg_std_probadistance_consecutives_llava(
    avgs,
    stds,
    xaxis,
    fontsize=30,
    font_offset=15,
    markersize=16,
    save_path=None,
    facecolor="lavender",
    plot_name="",
    labels=[""],
    skip_pp=False,
    per_modality=False,
    keys_to_remove=[],
):

    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams["lines.linewidth"] = 4.0
    plt.rcParams["font.family"] = "serif"  # Times New Roman serif
    plt.figure(figsize=(20, 15))
    ax = plt.axes()
    ax.set_facecolor(facecolor)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    color_idx = 0

    for i, key in enumerate(avgs.keys()):

        if key in keys_to_remove:
            continue

        avg = avgs[key]
        std = stds[key]

        n = key
        if per_modality:

            for j in range(len(avg[0][0])):

                lab = labels[0][j]
                averages = np.array([a[0][j] for a in avg])
                std_devs = np.array([a[0][j] for a in std])

                xaxis_ = xaxis[: len(averages)]

                if not skip_pp:
                    color = colors[color_idx]
                    color_idx += 1
                    if color_idx >= len(colors):
                        color_idx = 0
                    plt.plot(
                        xaxis_,
                        averages,
                        marker="o",
                        linestyle="--",
                        markersize=markersize,
                        label=f"{lab}-P (n {n})",
                        color=color,
                    )
                    plt.fill_between(
                        xaxis_,
                        averages - std_devs,
                        averages + std_devs,
                        alpha=0.2,
                        color=color,
                    )
                    plt.axhline(y=np.mean(averages), linestyle="--", color=color)

            for j in range(len(avg[0][1])):
                averages = np.array([a[1][j] for a in avg])
                std_devs = np.array([a[1][j] for a in std])

                xaxis_ = xaxis[: len(averages)]

                lab = labels[1][j]
                color = colors[color_idx]
                color_idx += 1
                if color_idx >= len(colors):
                    color_idx = 0
                plt.plot(
                    xaxis_,
                    averages,
                    marker="^",
                    linestyle="--",
                    markersize=markersize,
                    label=f"{lab}-T (n {n})",
                    color=color,
                )
                plt.fill_between(
                    xaxis_,
                    averages - std_devs,
                    averages + std_devs,
                    alpha=0.15,
                    color=color,
                )
                plt.axhline(y=np.mean(averages), linestyle="--", color=color)

        else:

            averages = np.array([a[0] for a in avg])
            std_devs = np.array([a[0] for a in std])

            xaxis_ = xaxis[: len(averages)]

            if not skip_pp:
                color = colors[color_idx]
                color_idx += 1
                if color_idx >= len(colors):
                    color_idx = 0
                plt.plot(
                    xaxis_,
                    averages,
                    marker="o",
                    linestyle="--",
                    markersize=markersize,
                    label=f"P (n {n})",
                    color=color,
                )
                plt.fill_between(
                    xaxis_,
                    averages - std_devs,
                    averages + std_devs,
                    alpha=0.2,
                    color=color,
                )
                plt.axhline(y=np.mean(averages), linestyle="--", color=color)

            averages = np.array([a[1] for a in avg])
            std_devs = np.array([a[1] for a in std])

            color = colors[color_idx]
            color_idx += 1
            if color_idx >= len(colors):
                color_idx = 0
            plt.plot(
                xaxis_,
                averages,
                marker="^",
                linestyle="--",
                markersize=markersize,
                label=f"T (n {n})",
                color=color,
            )
            plt.fill_between(
                xaxis_,
                averages - std_devs,
                averages + std_devs,
                alpha=0.15,
                color=color,
            )
            plt.axhline(y=np.mean(averages), linestyle="--", color=color)

    plt.ylabel("Distance", fontsize=fontsize + font_offset)
    plt.xlabel("Layers", fontsize=fontsize + font_offset)
    plt.legend(fontsize=fontsize + font_offset)

    plt.grid(color="white", linewidth=4)

    plt.xticks(fontsize=fontsize + font_offset)
    plt.yticks(fontsize=fontsize + font_offset)

    if save_path is not None:
        save_plot_path = os.path.join(save_path, f"{plot_name}.jpg")
        plt.savefig(save_plot_path, bbox_inches="tight")
        print(f"saved at {save_plot_path}")


results_dir = "/data/mshukor/logs/DePlam"
skip_last_txt_tokens = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="opt")
    parser.add_argument("--layer_norm", action="store_true")
    parser.add_argument("--mode", type=str, default="same")
    parser.add_argument("--lens", type=str, default="logit")
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

        text_model = "facebook/opt-6.7b"
        model_text = OPTForCausalLM.from_pretrained(
            text_model, torch_dtype=torch.float16, local_files_only=True
        )
        lm_head = model_text.lm_head
        layer_norm = model_text.model.decoder.final_layer_norm

        save_results_dir = "/data/mshukor/logs/DePlam/results/plots/opt"
        folders = [
            "ePALM_L_video_caption_l0_1_qsformerl10_timesformer",
            "ePALM_caption_qformerl10_vitl",
            "ePALM_L_audio_caption_l0_1_qsformerl10_ast",
        ]

    elif "vicuna" in args.model:

        text_model = "lmsys/vicuna-7b-v1.5"
        model_text = LlamaForCausalLM.from_pretrained(
            text_model, torch_dtype=torch.float16
        )
        lm_head = model_text.lm_head
        layer_norm = model_text.model.norm

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

        text_model = "/data/mshukor/pretrained_models/llama/llamav2_7b_hf"
        model_text = LlamaForCausalLM.from_pretrained(
            text_model, torch_dtype=torch.float16
        )
        lm_head = model_text.lm_head
        layer_norm = model_text.model.norm

    elif "llava" in args.model:

        text_model = "lmsys/vicuna-7b-v1.5"
        print(f"Reading {text_model} ...")

        model_text = LlamaForCausalLM.from_pretrained(
            text_model, torch_dtype=torch.float16
        )
        lm_head = model_text.lm_head
        layer_norm = model_text.model.norm

        save_results_dir = "/data/mshukor/logs/DePlam/results/plots/llava"
        prompt_len = 576

        if "qformernoptllavafrozen1round" in args.model:

            results_dir = "/data/mshukor/logs/DePlam/llava/llava_v1_5_qformer/hidden_states_1round/"
            mdl = "qformernoptllavafrozen1round"
            prompt_len = 32

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

    if not args.layer_norm:
        layer_norm = None
    else:
        layer_norm = layer_norm.float()
        suffix += "ln"

    if "consecutive" in args.mode:
        suffix += "_consec"

    if args.noise_baseline:
        suffix += "_noise"

    print(f"use LN: {layer_norm}, suffix:{suffix}, mode: {args.mode}")

    new_labels = labels

    index_img = -1
    index_txt = -1

    if "llava" in args.model:

        print(prompt_len)
        file_name = f"all_hidden_states.pth"
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
            discard_first_text=True,
        )

        if "consecutive" in args.mode:
            print(f"mode: {args.mode}")
            intervals = [1, 8]
            plot_layers = [0, 2, 4, 8, 12, 16, 20, 24, 28, 31]
            num_examples = 80

            all_avg, all_std = compute_proba_distance_from_hs_consecutive_across_layers(
                all_embed_img,
                all_embed_txt,
                new_labels,
                plot_layers,
                lm_head=lm_head.float(),
                num_examples=num_examples,
                skip_pp=False,
                layer_norm=layer_norm,
                intervals=intervals,
                mode=args.lens,
            )

            plt_avg_std_probadistance_consecutives(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"{mdl}emb_probadist_layers_avg_mode{suffix}",
                save_path=save_results_dir,
                skip_pp=False,
            )

            plt_avg_std_probadistance_consecutives(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"{mdl}emb_probadist_layers_avg_mode{suffix}_skip_pp",
                save_path=save_results_dir,
                skip_pp=True,
            )

            (
                all_avg,
                all_std,
                labels_,
            ) = compute_proba_distance_from_hs_consecutive_across_layers(
                all_embed_img,
                all_embed_txt,
                new_labels,
                plot_layers,
                lm_head=lm_head.float(),
                num_examples=num_examples,
                skip_pp=False,
                per_modality=True,
                layer_norm=layer_norm,
                intervals=intervals,
                mode=args.lens,
            )

            plt_avg_std_probadistance_consecutives_llava(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"{mdl}emb_probadist_layers_avg_mode{suffix}_permodal",
                save_path=save_results_dir,
                skip_pp=False,
                labels=labels_,
                per_modality=True,
            )

            plt_avg_std_probadistance_consecutives_llava(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"{mdl}emb_probadist_layers_avg_mode{suffix}_permodal_skip_pp",
                save_path=save_results_dir,
                skip_pp=True,
                labels=labels_,
                per_modality=True,
            )

        elif "entropy" in args.mode:

            plot_layers = [0, 2, 4, 8, 16, 24, 32]
            num_examples = 80

            all_avg, all_std = compute_proba_entropy_from_hs_across_layers(
                all_embed_img,
                all_embed_txt,
                new_labels,
                plot_layers,
                lm_head=lm_head.float(),
                num_examples=num_examples,
                skip_pp=False,
                layer_norm=layer_norm,
                mode=args.lens,
            )

            plt_avg_std_probadistance(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"{mdl}emb_probaentropy_layers_avg_mode{suffix}",
                save_path=save_results_dir,
                skip_pp=False,
                solo=True,
                ylabel="Entropy",
            )

            plt_avg_std_probadistance(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"{mdl}emb_probaentropy_layers_avg_mode{suffix}_skip_pp",
                save_path=save_results_dir,
                skip_pp=True,
                solo=True,
                ylabel="Entropy",
            )

            all_avg, all_std, labels_ = compute_proba_entropy_from_hs_across_layers(
                all_embed_img,
                all_embed_txt,
                new_labels,
                plot_layers,
                lm_head=lm_head.float(),
                num_examples=num_examples,
                skip_pp=False,
                per_modality=True,
                layer_norm=layer_norm,
                mode=args.lens,
            )

            plt_avg_std_probadistance_llava(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"{mdl}emb_probaentropy_layers_avg_mode{suffix}_permodal",
                save_path=save_results_dir,
                skip_pp=False,
                labels=labels_,
                per_modality=True,
                solo=True,
                ylabel="Entropy",
            )

            plt_avg_std_probadistance_llava(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"{mdl}emb_probaentropy_layers_avg_mode{suffix}_permodal_skip_pp",
                save_path=save_results_dir,
                skip_pp=True,
                labels=labels_,
                per_modality=True,
                solo=True,
                ylabel="Entropy",
            )

        else:

            plot_layers = [0, 32]

            for k in [1, 5]:

                all_ids = compute_hist_from_hs_across_layers(
                    all_embed_img,
                    all_embed_txt,
                    new_labels,
                    plot_layers,
                    lm_head=lm_head.float(),
                    k=k,
                    layer_norm=layer_norm,
                    mode=args.lens,
                )

                for ids, layer in zip(all_ids, plot_layers):
                    all_img_ids, all_txt_ids = ids

                    # all_img_ids, all_txt_ids = all_img_ids.flatten().numpy().tolist(), all_txt_ids.flatten().numpy().tolist()

                    plot_histograms(
                        all_img_ids,
                        all_txt_ids,
                        labels=[""],
                        plot_name=f"{mdl}emb_hist_layers_layer{layer}_k{k}{suffix}",
                        save_path=save_results_dir,
                        bins=120,
                        max_xrange=33000,
                        max_yrange=300,
                    )
                    # plot_histograms(all_img_ids, all_txt_ids, labels=[''], plot_name=f'{mdl}emb_kde_layers_layer{layer}_k{k}{suffix}', save_path=save_results_dir, bins=120,
                    #                 max_xrange=33000, max_yrange=300, kde_plot=True)

            plot_layers = [0, 2, 4, 8, 16, 24, 32]
            num_examples = 80

            all_avg, all_std = compute_proba_distance_from_hs_across_layers(
                all_embed_img,
                all_embed_txt,
                new_labels,
                plot_layers,
                lm_head=lm_head.float(),
                num_examples=num_examples,
                skip_pp=False,
                layer_norm=layer_norm,
                mode=args.lens,
            )

            plt_avg_std_probadistance(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"{mdl}emb_probadist_layers_avg_mode{suffix}",
                save_path=save_results_dir,
                skip_pp=False,
            )

            plt_avg_std_probadistance(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"{mdl}emb_probadist_layers_avg_mode{suffix}_skip_pp",
                save_path=save_results_dir,
                skip_pp=True,
            )

            all_avg, all_std, labels_ = compute_proba_distance_from_hs_across_layers(
                all_embed_img,
                all_embed_txt,
                new_labels,
                plot_layers,
                lm_head=lm_head.float(),
                num_examples=num_examples,
                skip_pp=False,
                per_modality=True,
                layer_norm=layer_norm,
                mode=args.lens,
            )

            plt_avg_std_probadistance_llava(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"{mdl}emb_probadist_layers_avg_mode{suffix}_permodal",
                save_path=save_results_dir,
                skip_pp=False,
                labels=labels_,
                per_modality=True,
            )

            plt_avg_std_probadistance_llava(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"{mdl}emb_probadist_layers_avg_mode{suffix}_permodal_skip_pp",
                save_path=save_results_dir,
                skip_pp=True,
                labels=labels_,
                per_modality=True,
            )

    else:

        file_name = f"all_hidden_states.pth"
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
            skip_last_txt_tokens=skip_last_txt_tokens,
        )

        if "consecutive" in args.mode:
            print(f"mode: {args.mode}")
            intervals = [1, 8]
            plot_layers = [0, 2, 4, 8, 12, 16, 20, 24, 28, 31]
            num_examples = 80

            all_avg, all_std = compute_proba_distance_from_hs_consecutive_across_layers(
                all_embed_img,
                all_embed_txt,
                new_labels,
                plot_layers,
                lm_head=lm_head.float(),
                num_examples=num_examples,
                skip_pp=False,
                layer_norm=layer_norm,
                intervals=intervals,
                mode=args.lens,
            )

            plt_avg_std_probadistance_consecutives(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"emb_probadist_layers_avg_mode{suffix}",
                save_path=save_results_dir,
                skip_pp=False,
            )

            plt_avg_std_probadistance_consecutives(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"emb_probadist_layers_avg_mode{suffix}_skip_pp",
                save_path=save_results_dir,
                skip_pp=True,
            )

            (
                all_avg,
                all_std,
                labels_,
            ) = compute_proba_distance_from_hs_consecutive_across_layers(
                all_embed_img,
                all_embed_txt,
                new_labels,
                plot_layers,
                lm_head=lm_head,
                num_examples=num_examples,
                skip_pp=False,
                per_modality=True,
                layer_norm=layer_norm,
                intervals=intervals,
                mode=args.lens,
            )

            plt_avg_std_probadistance_consecutives(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"emb_probadist_layers_avg_mode{suffix}_permodal",
                save_path=save_results_dir,
                skip_pp=False,
                labels=labels,
                per_modality=True,
            )

            plt_avg_std_probadistance_consecutives(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"emb_probadist_layers_avg_mode{suffix}_permodal_skip_pp",
                save_path=save_results_dir,
                skip_pp=True,
                labels=labels,
                per_modality=True,
            )
        elif "entropy" in args.mode:

            plot_layers = [0, 2, 4, 8, 16, 24, 32]
            num_examples = 80

            all_avg, all_std = compute_proba_entropy_from_hs_across_layers(
                all_embed_img,
                all_embed_txt,
                new_labels,
                plot_layers,
                lm_head=lm_head.float(),
                num_examples=num_examples,
                skip_pp=False,
                layer_norm=layer_norm,
                mode=args.lens,
            )

            plt_avg_std_probadistance(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"emb_probaentropy_layers_avg_mode{suffix}",
                save_path=save_results_dir,
                skip_pp=False,
                solo=True,
                ylabel="Entropy",
            )

            plt_avg_std_probadistance(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"emb_probaentropy_layers_avg_mode{suffix}_skip_pp",
                save_path=save_results_dir,
                skip_pp=True,
                solo=True,
                ylabel="Entropy",
            )

            all_avg, all_std, labels_ = compute_proba_entropy_from_hs_across_layers(
                all_embed_img,
                all_embed_txt,
                new_labels,
                plot_layers,
                lm_head=lm_head,
                num_examples=num_examples,
                skip_pp=False,
                per_modality=True,
                layer_norm=layer_norm,
                mode=args.lens,
            )

            plt_avg_std_probadistance(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"emb_probaentropy_layers_avg_mode{suffix}_permodal",
                save_path=save_results_dir,
                skip_pp=False,
                labels=labels,
                per_modality=True,
                solo=True,
                ylabel="Entropy",
            )

            plt_avg_std_probadistance(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"emb_probaentropy_layers_avg_mode{suffix}_permodal_skip_pp",
                save_path=save_results_dir,
                skip_pp=True,
                labels=labels,
                per_modality=True,
                solo=True,
                ylabel="Entropy",
            )

        else:
            plot_layers = [0, 32]

            for k in [1, 5]:

                all_ids = compute_hist_from_hs_across_layers(
                    all_embed_img,
                    all_embed_txt,
                    new_labels,
                    plot_layers,
                    lm_head=lm_head.float(),
                    k=k,
                    layer_norm=layer_norm,
                    mode=args.lens,
                )

                for ids, layer in zip(all_ids, plot_layers):
                    all_img_ids, all_txt_ids = ids

                    plot_histograms(
                        all_img_ids,
                        all_txt_ids,
                        labels=[""],
                        plot_name=f"emb_hist_layers_layer{layer}_k{k}{suffix}",
                        save_path=save_results_dir,
                        bins=120,
                        max_xrange=33000,
                        max_yrange=300,
                    )


                all_ids, labels_ = compute_hist_from_hs_across_layers(
                    all_embed_img,
                    all_embed_txt,
                    new_labels,
                    plot_layers,
                    lm_head=lm_head.float(),
                    k=k,
                    layer_norm=layer_norm,
                    mode=args.lens,
                    per_modality=True,
                )

                for ids, layer in zip(all_ids, plot_layers):
                    all_img_ids, all_txt_ids = ids

                    plot_histograms(
                        all_img_ids,
                        all_txt_ids,
                        labels=labels,
                        plot_name=f"emb_hist_layers_layer{layer}_k{k}{suffix}_permodal",
                        save_path=save_results_dir,
                        bins=120,
                        max_xrange=33000,
                        max_yrange=300,
                    )

            plot_layers = [0, 2, 4, 8, 16, 24, 32]
            num_examples = 80

            all_avg, all_std = compute_proba_distance_from_hs_across_layers(
                all_embed_img,
                all_embed_txt,
                new_labels,
                plot_layers,
                lm_head=lm_head.float(),
                num_examples=num_examples,
                skip_pp=False,
                layer_norm=layer_norm,
                mode=args.lens,
            )

            plt_avg_std_probadistance(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"emb_probadist_layers_avg_mode{suffix}",
                save_path=save_results_dir,
                skip_pp=False,
            )

            plt_avg_std_probadistance(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"emb_probadist_layers_avg_mode{suffix}_skip_pp",
                save_path=save_results_dir,
                skip_pp=True,
            )

            all_avg, all_std, labels_ = compute_proba_distance_from_hs_across_layers(
                all_embed_img,
                all_embed_txt,
                new_labels,
                plot_layers,
                lm_head=lm_head,
                num_examples=num_examples,
                skip_pp=False,
                per_modality=True,
                layer_norm=layer_norm,
                mode=args.lens,
            )

            plt_avg_std_probadistance(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"emb_probadist_layers_avg_mode{suffix}_permodal",
                save_path=save_results_dir,
                skip_pp=False,
                labels=labels,
                per_modality=True,
            )

            plt_avg_std_probadistance(
                all_avg,
                all_std,
                plot_layers,
                plot_name=f"emb_probadist_layers_avg_mode{suffix}_permodal_skip_pp",
                save_path=save_results_dir,
                skip_pp=True,
                labels=labels,
                per_modality=True,
            )
