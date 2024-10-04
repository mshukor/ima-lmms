import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from utils import (read_hidden_states_from_folders,
                   read_hidden_states_from_folders_llava)


def concat_hidden_states(
    all_embed_img,
    all_embed_txt,
    image_layer=0,
    text_layer=1,
    labels=[0, 1, 2, 3, 4, 5],
    per_modal=False,
):

    embeds = []
    new_labels = []

    if per_modal:
        for ims, txts, lab in zip(all_embed_img, all_embed_txt, labels):
            im1 = [im[image_layer] for im in ims]

            if im1[0].ndim > 1:
                im1 = np.concatenate(im1).tolist()
            embeds.append(im1)
            new_labels.append([f"{lab}-P"] * len(im1))
            txt1 = [txt[text_layer] for txt in txts]

            if txt1[0].ndim > 1:
                txt1 = np.concatenate(txt1).tolist()

            embeds.append(txt1)
            new_labels.append([f"{lab}-T"] * len(txt1))
    else:
        all_embed_img_ = []
        all_embed_txt_ = []

        if len(labels) < 2:
            all_img, all_txt = [all_embed_img], [all_embed_txt]
        else:
            all_img, all_txt = all_embed_img, all_embed_txt

        for all_embed_img, all_embed_txt, lab in zip(all_img, all_txt, labels):

            for ims, txts in zip(all_embed_img, all_embed_txt):
                all_embed_img_ += ims
                all_embed_txt_ += txts

            im1 = [im[image_layer] for im in all_embed_img_]

            if im1[0].ndim > 1:
                im1 = np.concatenate(im1).tolist()
                print(len(im1), im1[0].shape)
            embeds.append(im1)
            new_labels.append([f"{lab}-P"] * len(im1))
            txt1 = [txt[text_layer] for txt in all_embed_txt_]

            if txt1[0].ndim > 1:
                txt1 = np.concatenate(txt1).tolist()
                print(len(txt1), txt1[0].shape)

            embeds.append(txt1)
            new_labels.append([f"{lab}-T"] * len(txt1))
    return embeds, new_labels


def tsne_embdes(embeds, labels):

    all_data = np.concatenate(embeds)
    all_data = all_data.astype(float)

    print(all_data.shape)

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embedded_data = tsne.fit_transform(all_data)

    return embedded_data, labels


def plot_scatter_tsne(
    embedded_data,
    labels,
    fontsize=14,
    font_offset=15,
    facecolor="lavender",
    markersize=85,
    title="",
    plot_name="",
    save_path=None,
):

    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams["lines.linewidth"] = 4.0
    plt.rcParams["font.family"] = "serif"  # Times New Roman serif

    plt.figure(figsize=(15, 15))
    ax = plt.axes()
    ax.set_facecolor(facecolor)

    new_embedded_data = []

    prev = 0

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_idx = 0

    new_lab = []
    for i, l in enumerate(labels):

        emb = embedded_data[prev : prev + len(l)]
        lab = l[0]

        new_embedded_data.append(emb)
        new_lab.append(lab)

        prev = prev + len(l)

        if "T" in lab:
            marker = "^"
        else:
            marker = "o"

        color = colors[color_idx]
        color_idx += 1
        if color_idx >= len(colors):
            color_idx = 0

        plt.scatter(
            emb[:, 0], emb[:, 1], s=markersize, label=lab, marker=marker, color=color
        )

    ## matchine lines
    lab_to_idx = {l: i for i, l in enumerate(new_lab)}
    print(lab_to_idx)
    p_points, t_points = [], []

    processed_labs = []
    for lab in new_lab:
        if lab not in processed_labs:
            if "T" in lab:
                processed_labs += lab
                t_points.append(new_embedded_data[lab_to_idx[lab]])

                lab2 = lab.replace("T", "P")
                processed_labs += lab2
                p_points.append(new_embedded_data[lab_to_idx[lab2]])

            else:
                processed_labs += lab
                p_points.append(new_embedded_data[lab_to_idx[lab]])

                lab2 = lab.replace("P", "T")
                processed_labs += lab2
                t_points.append(new_embedded_data[lab_to_idx[lab2]])

    p_points, t_points = np.concatenate(p_points), np.concatenate(t_points)

    # Draw lines between pairs of points in the clusters
    for p1, p2 in zip(p_points, t_points):

        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", alpha=0.5, linewidth=0.5)

    plt.xticks(fontsize=fontsize + font_offset)
    plt.yticks(fontsize=fontsize + font_offset)

    plt.legend(fontsize=fontsize + font_offset)

    plt.grid(color="white", linewidth=4)

    if save_path is not None:
        save_plot_path = os.path.join(save_path, f"{plot_name}.jpg")
        plt.savefig(save_plot_path, bbox_inches="tight")
        print(f"saved at {save_plot_path}")

    # plt.show()
    plt.close()


def plot_scatter_tsne_hist(
    embedded_data,
    labels,
    fontsize=14,
    font_offset=15,
    facecolor="lavender",
    markersize=75,
    title="",
    plot_name="",
    save_path=None,
    feat_dim=0,
):

    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams["lines.linewidth"] = 4.0
    plt.rcParams["font.family"] = "serif"  # Times New Roman serif

    plt.figure(figsize=(15, 15))
    ax = plt.axes()
    ax.set_facecolor(facecolor)

    new_embedded_data = []

    prev = 0

    new_lab = []
    for i, l in enumerate(labels):

        emb = embedded_data[prev : prev + len(l)]
        emb = emb.astype(float)
        lab = l[0]

        new_embedded_data.append(emb)
        new_lab.append(lab)

        prev = prev + len(l)

        sns.kdeplot(emb[:, feat_dim], label=lab, alpha=0.3, fill=True)

    plt.ylabel("")

    plt.xticks(fontsize=fontsize + font_offset)
    plt.yticks(fontsize=fontsize + font_offset)

    plt.legend(fontsize=fontsize + font_offset)

    plt.grid(color="white", linewidth=4)

    if save_path is not None:
        save_plot_path = os.path.join(save_path, f"{plot_name}.jpg")
        plt.savefig(save_plot_path, bbox_inches="tight")
        print(f"saved at {save_plot_path}")

    plt.close()


results_dir = "/data/mshukor/logs/DePlam"
skip_last_txt_tokens = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="opt")
    parser.add_argument("--mode", type=str, default="inter")
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

        if "qformernoptllavafrozen1round" in args.model:

            results_dir = "/data/mshukor/logs/DePlam/llava/llava_v1_5_qformer/hidden_states_1round/"
            mdl = "qformernoptllavafrozen1round"
            prompt_len = 32

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

        if "alltok" in args.mode and "inside" in args.mode:
            results_dir = results_dir.replace(
                "hidden_states_1round", "hidden_states_1round_atts"
            )

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

    if "alltok" in args.mode:
        index_img = -1
        index_txt = -1
        suffix += f"{args.mode}_sim"
    else:
        index_img = None
        index_txt = None

    intervals = [1, 8]

    layer_indices = [0, 1, 4, 8, 16, 24, 31, 32]

    if "inside" in args.mode:
        # hidden_states_key = ['intermediate_hidden_preatts', 'intermediate_hidden_atts', 'intermediate_hidden_atts_res', 'intermediate_hidden_prefc',
        #                     'intermediate_hidden_act', 'intermediate_hidden_fc2', 'intermediate_hidden_fc2_res']

        hidden_states_key = [
            "intermediate_hidden_atts",
            "intermediate_hidden_act",
            "intermediate_hidden_fc2",
        ]

        if "llava" in args.mode and "alltok" in args.mode:
            hidden_states_key = ["intermediate_hidden_atts"]

        if "alltok" not in args.mode:
            prompt_len = 1  # we don't store all tokens inside
        if "alltok" in args.mode:
            suffix += "_alltok"
        layer_indices.pop(0)

    else:
        hidden_states_key = ["hidden_states"]

    if "llava" in args.model:

        file_name = args.file_name
        if "1k" in file_name:
            suffix += "_1k"

        if "epochs" in args.mode:
            avgs = []
            stds = []
            all_embed_img_, all_embed_txt_ = [], []

            epochs = ["0000", "1500", "3000", "4500", "6000"]
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
                    discard_first_text=True,
                    hidden_states_key=hidden_states_key,
                )
                suffix_ = suffix + f"epoch{e}"

                all_embed_img_.append(all_embed_img)
                all_embed_txt_.append(all_embed_txt)

                for layer_index in layer_indices:
                    print(layer_index)
                    embeds, new_labels_ = concat_hidden_states(
                        all_embed_txt,
                        all_embed_txt,
                        image_layer=layer_index,
                        text_layer=layer_index,
                        labels=[""],
                    )

                    try:
                        embedded_data, true_labels = tsne_embdes(embeds, new_labels_)

                        plot_scatter_tsne(
                            embedded_data,
                            true_labels,
                            save_path=save_results_dir,
                            plot_name=f"{mdl}_{suffix_}hsl{layer_index}_tsne_true_labels",
                        )

                        feat_dim = 0
                        plot_scatter_tsne_hist(
                            embedded_data,
                            true_labels,
                            save_path=save_results_dir,
                            plot_name=f"{mdl}_{suffix_}hsl{layer_index}_tsne_true_labels_hist{feat_dim}",
                            feat_dim=feat_dim,
                        )
                    except:
                        print("error in tsne", layer_index, e)

            suffix_ = "all_epochs"
            epochs_ = [f"step {s}" for s in epochs]

            embeds, new_labels_ = concat_hidden_states(
                all_embed_img_,
                all_embed_txt_,
                image_layer=layer_index,
                text_layer=layer_index,
                labels=epochs_,
            )

            embedded_data, true_labels = tsne_embdes(embeds, new_labels_)

            plot_scatter_tsne(
                embedded_data,
                true_labels,
                save_path=save_results_dir,
                plot_name=f"{mdl}_{suffix_}hsl{layer_index}_tsne_true_labels",
            )

            feat_dim = 0
            plot_scatter_tsne_hist(
                embedded_data,
                true_labels,
                save_path=save_results_dir,
                plot_name=f"{mdl}_{suffix_}hsl{layer_index}_tsne_true_labels_hist{feat_dim}",
                feat_dim=feat_dim,
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
                discard_first_text=True,
                hidden_states_key=hidden_states_key,
            )

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

                for layer_index in layer_indices:
                    print("layer_index", layer_index, hidden_states_key)
                    embeds, new_labels_ = concat_hidden_states(
                        all_embed_img_,
                        all_embed_txt_,
                        image_layer=layer_index,
                        text_layer=layer_index,
                        labels=[""],
                    )

                    embedded_data, true_labels = tsne_embdes(embeds, new_labels_)

                    plot_scatter_tsne(
                        embedded_data,
                        true_labels,
                        save_path=save_results_dir,
                        plot_name=f"{mdl}_{suffix_}hsl{layer_index}_tsne_true_labels",
                    )

                    feat_dim = 0
                    plot_scatter_tsne_hist(
                        embedded_data,
                        true_labels,
                        save_path=save_results_dir,
                        plot_name=f"{mdl}_{suffix_}hsl{layer_index}_tsne_true_labels_hist{feat_dim}",
                        feat_dim=feat_dim,
                    )
    else:

        file_name = args.file_name
        if "1k" in file_name:
            suffix += "_1k"

        if "epochs" in args.mode:
            avgs = []
            stds = []
            all_embed_img_, all_embed_txt_ = [], []

            for e in [-1, 0, 1, 2]:
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

                all_embed_img_.append(all_embed_img)
                all_embed_txt_.append(all_embed_txt)

                for layer_index in [0, 1]:
                    print(layer_index)
                    embeds, new_labels_ = concat_hidden_states(
                        all_embed_img,
                        all_embed_txt,
                        image_layer=layer_index,
                        text_layer=layer_index,
                        labels=[""],
                    )

                    embedded_data, true_labels = tsne_embdes(embeds, new_labels_)

                    plot_scatter_tsne(
                        embedded_data,
                        true_labels,
                        save_path=save_results_dir,
                        plot_name=f"{suffix_}hsl{layer_index}_tsne_true_labels",
                    )

                    feat_dim = 0
                    plot_scatter_tsne_hist(
                        embedded_data,
                        true_labels,
                        save_path=save_results_dir,
                        plot_name=f"{suffix_}hsl{layer_index}_tsne_true_labels_hist{feat_dim}",
                        feat_dim=feat_dim,
                    )

            suffix_ = "all_epochs"
            epochs_ = ["ep 0", "ep 1", "ep 2", "ep 3"]

            embeds, new_labels_ = concat_hidden_states(
                all_embed_img_,
                all_embed_txt_,
                image_layer=layer_index,
                text_layer=layer_index,
                labels=epochs_,
            )

            embedded_data, true_labels = tsne_embdes(embeds, new_labels_)

            plot_scatter_tsne(
                embedded_data,
                true_labels,
                save_path=save_results_dir,
                plot_name=f"{suffix_}hsl{layer_index}_tsne_true_labels",
            )

            feat_dim = 0
            plot_scatter_tsne_hist(
                embedded_data,
                true_labels,
                save_path=save_results_dir,
                plot_name=f"{suffix_}hsl{layer_index}_tsne_true_labels_hist{feat_dim}",
                feat_dim=feat_dim,
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

                print(all_embed_img_[0][0][1].shape)

                for layer_index in layer_indices:
                    print(layer_index)
                    embeds, new_labels_ = concat_hidden_states(
                        all_embed_img_,
                        all_embed_txt_,
                        image_layer=layer_index,
                        text_layer=layer_index,
                        labels=[""],
                    )

                    embedded_data, true_labels = tsne_embdes(embeds, new_labels_)

                    plot_scatter_tsne(
                        embedded_data,
                        true_labels,
                        save_path=save_results_dir,
                        plot_name=f"{suffix_}hsl{layer_index}_tsne_true_labels",
                    )

                    feat_dim = 0
                    plot_scatter_tsne_hist(
                        embedded_data,
                        true_labels,
                        save_path=save_results_dir,
                        plot_name=f"{suffix_}hsl{layer_index}_tsne_true_labels_hist{feat_dim}",
                        feat_dim=feat_dim,
                    )

                    embeds, new_labels_ = concat_hidden_states(
                        all_embed_img_,
                        all_embed_txt_,
                        image_layer=layer_index,
                        text_layer=layer_index,
                        labels=labels,
                        per_modal=True,
                    )

                    embedded_data, true_labels = tsne_embdes(embeds, new_labels_)

                    plot_scatter_tsne(
                        embedded_data,
                        true_labels,
                        save_path=save_results_dir,
                        plot_name=f"{suffix_}hsl{layer_index}_tsne_true_labels_permodal",
                    )

                    feat_dim = 0
                    plot_scatter_tsne_hist(
                        embedded_data,
                        true_labels,
                        save_path=save_results_dir,
                        plot_name=f"{suffix_}hsl{layer_index}_tsne_true_labels_hist{feat_dim}_permodal",
                        feat_dim=feat_dim,
                    )
