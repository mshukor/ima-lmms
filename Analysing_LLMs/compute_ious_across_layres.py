import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


def calculate_iou(matrix1, matrix2, pruned_weights=False):
    if not pruned_weights:
        matrix1, matrix2 = ~matrix1, ~matrix2

    intersection = np.logical_and(matrix1, matrix2)
    union = np.logical_or(matrix1, matrix2)

    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def compute_ious_acrosslayers(
    mask_1, layer_name="fc2", mode="subsequent", layer_interval=2, base_layer_index=0
):

    ious = []
    if mode == "subsequent":
        L = len(mask_1)
        for i in tqdm(range(L)):
            matrix1 = mask_1[i][layer_name].numpy()
            if i + layer_interval >= L:
                break
            matrix2 = mask_1[i + layer_interval][layer_name].numpy()

            iou = calculate_iou(matrix1, matrix2)
            ious.append(iou)
    return ious


def visualize_intersection(matrix1, matrix2):

    w, h = matrix2.shape
    x_label = "Input dimension"
    y_label = "Output dimension"
    if w > h:
        matrix1 = matrix1.transpose()
        matrix2 = matrix2.transpose()
        y_label = "Input dimension"
        x_label = "Output dimension"

    intersection_matrix = np.logical_and(matrix1, matrix2)

    plt.figure(figsize=(60, 20))

    plt.subplot(3, 1, 1)
    plt.title("Matrix 1")
    plt.imshow(matrix1, cmap="Blues", interpolation="none")
    plt.ylabel(y_label)

    plt.subplot(3, 1, 2)
    plt.title("Matrix 2")
    plt.imshow(matrix2, cmap="Greens", interpolation="none")
    plt.ylabel(y_label)

    plt.subplot(3, 1, 3)
    plt.title("Intersection")
    plt.imshow(intersection_matrix, cmap="Reds", interpolation="none")
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    # plt.show()

    iou = calculate_iou(matrix1, matrix2)
    print(f"IoU: {iou}")


def compute_mask_from_metric(metric_path, sparsity_ratio=0.5):

    print(f"pruning {metric_path}")
    metric = torch.load(metric_path)
    nb_layers = len(metric)
    W_masks = {i: {} for i in range(nb_layers)}

    for i in tqdm(range(nb_layers)):
        for layer_name in layer_names:

            W_metric = metric[i][layer_name]

            W_mask = torch.zeros_like(W_metric) == 1
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:, : int(W_metric.shape[1] * sparsity_ratio)]
            W_mask.scatter_(1, indices, True)

            W_masks[i].update({layer_name: W_mask.detach().cpu()})

    return W_masks


def compute_ious_from_metrices(
    model_name_dirs1, model_name_dirs2, layer_names, masks_dir, mask_name, sparsity=0.5
):
    all_ious = {k: [] for k in layer_names}
    all_ious["avg"] = []
    for model_name_dir1, model_name_dir2 in zip(model_name_dirs1, model_name_dirs2):

        file_path1 = os.path.join(masks_dir, model_name_dir1, mask_name)
        file_path2 = os.path.join(masks_dir, model_name_dir2, mask_name)

        mask_1 = compute_mask_from_metric(file_path1, sparsity_ratio=sparsity)
        mask_2 = compute_mask_from_metric(file_path2, sparsity_ratio=sparsity)

        tmp = []
        for layer_name in layer_names:
            print(layer_name)
            ious_m = compute_ious_acrossmasks(mask_1, mask_2, layer_name=layer_name)
            all_ious[layer_name].append(ious_m)
            tmp.append(ious_m)

        all_ious["avg"].append(np.mean(np.array(tmp), axis=0).tolist())
    return all_ious


def compute_ious_acrossmasks(mask_1, mask_2, layer_name="fc2", startl=0):

    ious = []
    for i in tqdm(range(len(mask_1))):
        if i >= startl:
            if layer_name in mask_1[i]:
                matrix1 = mask_1[i][layer_name].numpy()
                matrix2 = mask_2[i][layer_name].numpy()

                iou = calculate_iou(matrix1, matrix2)
                ious.append(iou)
            else:
                print(f"skip {i} {layer_name}")
    return ious


def compute_ious_dirs(
    model_name_dirs1, model_name_dirs2, layer_names, masks_dir, mask_name, startl=0
):
    all_ious = {k: [] for k in layer_names}
    all_ious["avg"] = []
    for model_name_dir1, model_name_dir2 in zip(model_name_dirs1, model_name_dirs2):

        file_path1 = os.path.join(masks_dir, model_name_dir1, mask_name)
        file_path2 = os.path.join(masks_dir, model_name_dir2, mask_name)

        mask_1 = torch.load(file_path1)
        mask_2 = torch.load(file_path2)

        tmp = []
        for layer_name in layer_names:
            print(layer_name)
            ious_m = compute_ious_acrossmasks(
                mask_1, mask_2, layer_name=layer_name, startl=startl
            )
            all_ious[layer_name].append(ious_m)
            tmp.append(ious_m)

        all_ious["avg"].append(np.mean(np.array(tmp), axis=0).tolist())
    return all_ious


def compute_ious_names(model_name_1, model_name_2, layer_names, masks_dir, startl=0):
    all_ious = {k: [] for k in layer_names}
    all_ious["avg"] = []
    labels = [True for i in range(len(model_name_1))]
    for i, (model_name_dir1, model_name_dir2) in enumerate(
        zip(model_name_1, model_name_2)
    ):

        file_path1 = os.path.join(masks_dir, model_name_dir1)
        file_path2 = os.path.join(masks_dir, model_name_dir2)

        if os.path.exists(file_path1) and os.path.exists(file_path2):
            mask_1 = torch.load(file_path1)
            mask_2 = torch.load(file_path2)

            tmp = []
            for layer_name in layer_names:
                print(layer_name)
                ious_m = compute_ious_acrossmasks(
                    mask_1, mask_2, layer_name=layer_name, startl=startl
                )
                all_ious[layer_name].append(ious_m)
                tmp.append(ious_m)

            all_ious["avg"].append(np.mean(np.array(tmp), axis=0).tolist())
        else:
            labels[i] = False
            print(f"not foun {file_path1} or {file_path2}")
    return all_ious, labels


def save_to_dict(d, all_ious, mask_names_1, mask_names_2, labels_ok, layer_names):
    mask_names_1_ok = [
        mask_names_1[i] for i in range(len(mask_names_1)) if labels_ok[i]
    ]
    mask_names_2_ok = [
        mask_names_2[i] for i in range(len(mask_names_2)) if labels_ok[i]
    ]
    for layer_name in layer_names:
        ious = all_ious[layer_name]
        for n1, n2, iou in zip(mask_names_1_ok, mask_names_2_ok, ious):
            key = f"{n1}_VS_{n2}"
            value = iou
            d[layer_name].update({key: value})
    return d


lname_2_title = {
    "avg": "All layer types",
    "fc1": "FFN (down proj)",
    "fc2": "FFN (up proj)",
    "self_attn.v_proj": "SA (values)",
    "self_attn.k_proj": "SA (keys)",
    "self_attn.q_proj": "SA (queries)",
    "self_attn.out_proj": "SA (out proj)",
    "mlp.gate_proj": "FFN (gate)",
    "mlp.down_proj": "FFN (down proj)",
    "mlp.up_proj": "FFN (up proj)",
    "self_attn.o_proj": "SA (out proj)",
}


encoder_2_encodername = {
    "_clip_encoder": "CLIP",
    "_mae_encoder": "MAE",
    "_vit_encoder": "ViT",
}

modality_2_modalityname = {
    "image": "I",
    "video": "V",
    "audio": "A",
}


def get_ious_from_dict(all_ious_dict, mask_names_1, mask_names_2, labels, layer_names):

    all_ious_ = {l: [] for l in layer_names}
    labels_ = {l: [] for l in layer_names}

    for layer_name in layer_names:
        all_ious = all_ious_dict[layer_name]

        for n1, n2, label in zip(mask_names_1, mask_names_2, labels):
            key = f"{n1}_VS_{n2}"
            if key in all_ious:
                value = all_ious[key]
                all_ious_[layer_name].append(value)
                labels_[layer_name].append(label)

    return all_ious_, labels_


def plot_ious(
    layer_names,
    all_ious,
    labels_toshow,
    markersize=12,
    fontsize=25,
    font_offset=15,
    data_end_offset=None,
    sparsity="",
    save_path=None,
    facecolor="lavender",
    encoder="",
    plot_name="",
    keys_to_remove_from_title=[],
    modality=None,
):
    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams["lines.linewidth"] = 4.0
    plt.rcParams["font.family"] = "serif"  # Times New Roman serif

    for layer_name in layer_names_:

        ious = all_ious[layer_name]

        labels = labels_toshow[layer_name]  # labels_toshow[layer_name]

        plt.figure(figsize=(20, 15))
        ax = plt.axes()
        ax.set_facecolor(facecolor)

        for ious_m, label in zip(ious, labels):
            if data_end_offset is not None:
                ious_m_ = ious_m[:data_end_offset]
            else:
                ious_m_ = ious_m

            plt.plot(
                ious_m_, label=label, marker="o", linestyle="--", markersize=markersize
            )
            plt.axhline(y=np.mean(ious_m_), linestyle="--")

        plt.ylabel("Masks IoU", fontsize=fontsize + font_offset)
        plt.xlabel("Transformer Layer", fontsize=fontsize + font_offset)
        plt.legend(fontsize=fontsize + font_offset)
        s = sparsity.replace("_", ".")

        if encoder and "encoder" not in keys_to_remove_from_title:
            enc = f"{encoder_2_encodername[encoder]}, "
        else:
            enc = ""

        if modality is not None and "modality" not in keys_to_remove_from_title:
            mod = modality_2_modalityname[modality]
            enc = f"{enc}{mod}, "

        # if encoder:
        title = lname_2_title[layer_name] + f" ({enc}Sparsity: {s})"
        # plt.title(title, fontsize=fontsize+font_offset)

        plt.suptitle(title, fontsize=fontsize + font_offset, y=0.95)

        plt.grid(color="white", linewidth=4)

        if save_path is not None:
            name = "_".join(title.split(" "))
            save_plot_path = os.path.join(save_path, f"{name}_{plot_name}.jpg")
            plt.savefig(save_plot_path, bbox_inches="tight")
            print(f"saved at {save_plot_path}")

        # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="opt")
    parser.add_argument("--layer_norm", action="store_true")

    args = parser.parse_args()

    if args.model == "vicuna":

        masks_dir = "/data/mshukor/logs/DePlam/masks/vicuna"
        layer_names = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]  # llamav2
        text_model = "llamav2vicuna_"
        encoders = [""]  # which is with vit encdoers
        sparsities = ["0_5"]
        save_results_dir = "/data/mshukor/logs/DePlam/results/plots/vicuna"
        plot_name_text_model = "_llamav2vicuna"

    elif args.model == "opt":

        masks_dir = "/data/mshukor/logs/DePlam/masks"
        layer_names = [
            "fc1",
            "fc2",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.out_proj",
        ]  # opt
        text_model = ""
        encoders = ["", "_clip_encoder", "_mae_encoder", "_vit_encoder"]
        sparsities = ["0_3", "0_5"]
        save_results_dir = "/data/mshukor/logs/DePlam/results/plots/opt"
        plot_name_text_model = "_opt"

    elif args.model == "opt":
        # masks_dir = '/data/mshukor/logs/DePlam/masks/llamav2'
        masks_dir = "/net/sister/mshukor/logs/masks/llamav2"
        layer_names = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]  # llamav2
        text_model = "llamav2_"
        encoders = [""]  # which is with vit encdoers
        sparsities = ["0_3", "0_5"]
        save_results_dir = "/data/mshukor/logs/DePlam/results/plots/llamav2"
        plot_name_text_model = "_llamav2"

    elif "llava" in args.model:

        masks_dir = f"/data/mshukor/logs/DePlam/masks/llava/{args.model}"
        layer_names = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]  # llamav2
        text_model = f"llava_"
        encoders = [""]  # which is with vit encdoers
        sparsities = ["0_5"]
        save_results_dir = "/data/mshukor/logs/DePlam/results/plots/llava"
        plot_name_text_model = f"_{args.model}"

    else:
        raise NotImplemented

    plot_name = "_vs_T" + plot_name_text_model

    all_ious_dict = {k: {} for k in layer_names + ["avg"]}

    plt.rcParams.update({"font.size": 20})
    plt.rcParams["lines.linewidth"] = 3.0

    for sparsity in sparsities:
        for encoder in encoders:

            if "llava" in args.model:
                mask_names_1 = [
                    f"masks_onlypromptwithans_all_{text_model}image{encoder}_masks_W_masks_s{sparsity}.pth",
                ]
                mask_names_2 = [
                    f"masks_onlytextwithans_all_{text_model}image{encoder}_masks_W_masks_s{sparsity}.pth",
                ]

                labels = [
                    "I vs T",
                ]

            else:
                mask_names_1 = [
                    f"masks_onlypromptwithans_all_{text_model}image{encoder}_masks_W_masks_s{sparsity}.pth",
                    f"masks_onlypromptwithans_all_{text_model}video{encoder}_masks_W_masks_s{sparsity}.pth",
                    f"masks_onlypromptwithans_all_{text_model}audio{encoder}_masks_W_masks_s{sparsity}.pth",
                ]

                mask_names_2 = [
                    f"masks_onlytextwithans_all_{text_model}image{encoder}_masks_W_masks_s{sparsity}.pth",
                    f"masks_onlytextwithans_all_{text_model}video{encoder}_masks_W_masks_s{sparsity}.pth",
                    f"masks_onlytextwithans_all_{text_model}audio{encoder}_masks_W_masks_s{sparsity}.pth",
                ]

                labels = [
                    "I vs T",
                    "V vs T",
                    "A vs T",
                ]

            print(encoder, sparsity)

            startl = 0
            all_ious, labels_ok = compute_ious_names(
                mask_names_1, mask_names_2, layer_names, masks_dir, startl=startl
            )
            labels_toshow = [labels[i] for i in range(len(labels)) if labels_ok[i]]

            labels_toshow = labels

            all_ious_dict = save_to_dict(
                all_ious_dict,
                all_ious,
                mask_names_1,
                mask_names_2,
                labels_ok,
                layer_names + ["avg"],
            )

            layer_names_ = ["avg"] + layer_names
            # layer_names_ = ['avg']

            all_ious_, labels_ = get_ious_from_dict(
                all_ious_dict, mask_names_1, mask_names_2, labels, layer_names_
            )
            plot_ious(
                layer_names_,
                all_ious_,
                labels_,
                data_end_offset=-1,
                sparsity=sparsity,
                save_path=save_results_dir,
                encoder=encoder,
                plot_name=plot_name,
            )
