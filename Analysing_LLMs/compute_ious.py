import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append("DePALM/")
from pruning.wanda_mm import compute_ex_mask


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


def compute_ious_acrossmasks(mask_1, mask_2, layer_name="fc2", startl=0):

    ious = []
    for i in range(len(mask_1)):
        if i >= startl:
            if layer_name in mask_1[i]:
                matrix1 = mask_1[i][layer_name].numpy()
                matrix2 = mask_2[i][layer_name].numpy()

                iou = calculate_iou(matrix1, matrix2)
                ious.append(iou)
            else:
                print(f"skip {i} {layer_name}")
    return ious


def compute_ious(file_path1, file_path2, layer_names, startl=0, computemask=False):
    all_ious = {k: [] for k in layer_names}
    all_ious["avg"] = []

    # try:
    if computemask:
        mask_1 = compute_mask(file_path1, ex_sparsity=1, layer_names=layer_names)
        mask_2 = compute_mask(file_path2, ex_sparsity=1, layer_names=layer_names)
    else:
        mask_1 = torch.load(file_path1)
        mask_2 = torch.load(file_path2)

    tmp = []
    for layer_name in layer_names:
        ious_m = compute_ious_acrossmasks(
            mask_1, mask_2, layer_name=layer_name, startl=startl
        )
        all_ious[layer_name].append(ious_m)
        tmp.append(ious_m)

    all_ious["avg"].append(np.mean(np.array(tmp), axis=0).tolist())
    # except:
    #     print(f"not foun {file_path1} or {file_path2}")

    return all_ious


def save_to_dict(d, all_ious, mask_names_1, mask_names_2, layer_names):
    for layer_name in layer_names:
        ious = all_ious[layer_name]
        for n1, n2, iou in zip(mask_names_1, mask_names_2, ious):
            key = f"{n1}_VS_{n2}"
            value = iou
            d[layer_name].update({key: value})
    return d


def check_already_computed(d, mask_names_1, mask_names_2, layer_names):

    for layer_name in layer_names:
        for n1, n2 in zip(mask_names_1, mask_names_2):
            key = f"{n1}_VS_{n2}"
            if key in d[layer_name]:
                return True

    return False


def modify_mask(mode, file_path1, file_path2, mask2_path, mask_name):

    if "without_mag" in mode:
        computemask = True
        if "0.3" in mask_name:
            mask2_path = os.path.join(mask2_path, "W_masks_mag_s0.3.pth")
        elif "0.5" in mask_name:
            mask2_path = os.path.join(mask2_path, "W_masks_mag_s0.5.pth")

        file_path1 = file_path1 + "!!!" + mask2_path
        file_path2 = file_path2 + "!!!" + mask2_path

    return file_path1, file_path2, computemask


def get_ious(
    folders,
    mask_dir,
    mask_type,
    mask_name,
    layer_names,
    all_ious_dict={},
    startl=0,
    save_path=None,
    folders_text=None,
    mode="",
    mask2_path=None,
    text_mask="masks_onlytextwithans",
):

    computemask = False

    if folders_text is not None:
        for i, folder1 in tqdm(enumerate(folders_text)):
            for j, folder2 in tqdm(enumerate(folders_text)):

                if j <= i:

                    file_path1 = os.path.join(mask_dir, folder1, text_mask, mask_name)
                    file_path2 = os.path.join(mask_dir, folder2, text_mask, mask_name)

                    mask_names_1 = [folder_2_name[folder1] + " Text"]
                    mask_names_2 = [folder_2_name[folder2] + " Text"]

                    if check_already_computed(
                        all_ious_dict, mask_names_1, mask_names_2, layer_names + ["avg"]
                    ):
                        continue

                    if "without_mag" in mode:
                        file_path1, file_path2, computemask = modify_mask(
                            mode, file_path1, file_path2, mask2_path, mask_name
                        )

                    print(
                        file_path1, file_path2, mask_names_1, mask_names_2, computemask
                    )

                    try:
                        all_ious = compute_ious(
                            file_path1,
                            file_path2,
                            layer_names,
                            startl=startl,
                            computemask=computemask,
                        )
                        all_ious_dict = save_to_dict(
                            all_ious_dict,
                            all_ious,
                            mask_names_1,
                            mask_names_2,
                            layer_names + ["avg"],
                        )
                    except RuntimeError:
                        print(
                            f"########################error with {file_path1}, {file_path2}"
                        )

    for i, folder1 in tqdm(enumerate(folders)):
        if folders_text is not None:
            for j, folder2 in tqdm(enumerate(folders_text)):

                file_path1 = os.path.join(mask_dir, folder1, mask_type, mask_name)
                file_path2 = os.path.join(mask_dir, folder2, text_mask, mask_name)

                mask_names_1 = [folder_2_name[folder1]]
                mask_names_2 = [folder_2_name[folder2] + " Text"]

                if check_already_computed(
                    all_ious_dict, mask_names_1, mask_names_2, layer_names + ["avg"]
                ):
                    continue

                if "without_mag" in mode:
                    file_path1, file_path2, computemask = modify_mask(
                        mode, file_path1, file_path2, mask2_path, mask_name
                    )

                print(file_path1, file_path2, mask_names_1, mask_names_2, computemask)

                try:
                    all_ious = compute_ious(
                        file_path1,
                        file_path2,
                        layer_names,
                        startl=startl,
                        computemask=computemask,
                    )
                    all_ious_dict = save_to_dict(
                        all_ious_dict,
                        all_ious,
                        mask_names_1,
                        mask_names_2,
                        layer_names + ["avg"],
                    )
                except RuntimeError:
                    print(
                        f"########################error with {file_path1}, {file_path2}"
                    )

        for j, folder2 in tqdm(enumerate(folders)):

            if j <= i:

                if mask_type is not None:
                    file_path1 = os.path.join(mask_dir, folder1, mask_type, mask_name)
                    file_path2 = os.path.join(mask_dir, folder2, mask_type, mask_name)
                else:
                    file_path1 = os.path.join(mask_dir, folder1, mask_name)
                    file_path2 = os.path.join(mask_dir, folder2, mask_name)

                mask_names_1 = [folder_2_name[folder1]]
                mask_names_2 = [folder_2_name[folder2]]

                if check_already_computed(
                    all_ious_dict, mask_names_1, mask_names_2, layer_names + ["avg"]
                ):
                    continue

                if "without_mag" in mode:
                    file_path1, file_path2, computemask = modify_mask(
                        mode, file_path1, file_path2, mask2_path, mask_name
                    )

                print(file_path1, file_path2, mask_names_1, mask_names_2)

                try:
                    all_ious = compute_ious(
                        file_path1,
                        file_path2,
                        layer_names,
                        startl=startl,
                        computemask=computemask,
                    )
                    all_ious_dict = save_to_dict(
                        all_ious_dict,
                        all_ious,
                        mask_names_1,
                        mask_names_2,
                        layer_names + ["avg"],
                    )
                except RuntimeError:
                    print(
                        f"########################error with {file_path1}, {file_path2}"
                    )

        if save_path is not None and i % 2 == 0:
            with open(save_path, "w") as f:
                json.dump(all_ious_dict, f)
            print(f"saved to {save_path}")

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(all_ious_dict, f)
        print(f"saved to {save_path}")

    return all_ious_dict


def get_ious_withtext(
    folders,
    mask_dir,
    mask_type,
    mask_name,
    layer_names,
    all_ious_dict={},
    startl=0,
    save_path=None,
):

    for i, folder1 in tqdm(enumerate(folders)):

        mask_type1 = "masks_onlypromptwithans"
        mask_type2 = "masks_onlytextwithans"

        file_path1 = os.path.join(mask_dir, folder1, mask_type1, mask_name)
        file_path2 = os.path.join(mask_dir, folder1, mask_type2, mask_name)

        mask_names_1 = [folder_2_name[folder1]]
        mask_names_2 = [mask_names_1[0] + " Text"]
        print(file_path1, file_path2, mask_names_1, mask_names_2)

        all_ious = compute_ious(file_path1, file_path2, layer_names, startl=startl)

        all_ious_dict = save_to_dict(
            all_ious_dict, all_ious, mask_names_1, mask_names_2, layer_names + ["avg"]
        )

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(all_ious_dict, f)
        print(f"saved to {save_path}")

    return all_ious_dict


def compute_mask(mask_path=None, ex_sparsity=0.015, layer_names=[]):

    # use_cache = model.model_text.config.use_cache
    # model.model_text.config.use_cache = False
    if "!!!" in mask_path:  # ablate
        mask_path1, mask_path2 = mask_path.split("!!!")
        W_masks1 = torch.load(mask_path1)
        if "random" in mask_path2:
            W_masks2 = None
        else:
            W_masks2 = torch.load(mask_path2)

        W_masks = compute_ex_mask(
            W_masks1, W_masks2, s=ex_sparsity, layer_names=layer_names
        )
    elif "???" in mask_path:
        print("compute intersection mask")
        mask_path1, mask_path2 = mask_path.split("???")
        W_masks1 = torch.load(mask_path1)
        if "random" in mask_path2:
            W_masks2 = None
        else:
            W_masks2 = torch.load(mask_path2)

        W_masks = compute_ex_mask(
            W_masks1,
            W_masks2,
            s=ex_sparsity,
            mode="intersection",
            layer_names=layer_names,
        )
    elif "!?!" in mask_path:
        print("compute union mask")
        mask_path1, mask_path2 = mask_path.split("!?!")
        W_masks1 = torch.load(mask_path1)
        if "random" in mask_path2:
            W_masks2 = None
        else:
            W_masks2 = torch.load(mask_path2)

        W_masks = compute_ex_mask(
            W_masks1, W_masks2, s=ex_sparsity, mode="union", layer_names=layer_names
        )

    else:
        W_masks = torch.load(mask_path)

    return W_masks

##### Vicuna

folders_vicuna = [
    "ePALM_caption_qformerl10_llamav2vicuna_vitl",
    "ePALM_gqa_qformerl10_llamav2vicuna_vitl",
    "ePALM_vqa_qformerl10_llamav2vicuna_vitl",
    "ePALM_okvqa_qformerl10_llamav2vicuna_vitl",
    "ePALM_L_video_caption_l0_1_qsformerl10_llamav2vicuna_timesformer",
    "ePALM_L_video_qa_msvd_l0_1_qsformerl10_llamav2vicuna_timesformer",
    "ePALM_L_audio_caption_l0_1_qsformerl10_llamav2vicuna_ast",
    "ePALM_L_clothoaqa_audio_qa_l0_1_qsformerl10_llamav2vicuna_ast",
]

vicuna_folder_2_name = {
    "ePALM_okvqa_qformerl10_llamav2vicuna_vitl": "OKVQA (ViT)",
    "ePALM_gqa_qformerl10_llamav2vicuna_vitl": "GQA (ViT)",
    "ePALM_vqa_qformerl10_llamav2vicuna_vitl": "VQA (ViT)",
    "ePALM_caption_qformerl10_llamav2vicuna_vitl": "COCO (ViT)",
    "ePALM_L_video_caption_l0_1_qsformerl10_llamav2vicuna_timesformer": "MSRVTT (ViT)",
    "ePALM_L_video_qa_msvd_l0_1_qsformerl10_llamav2vicuna_timesformer": "MSVD (ViT)",
    "ePALM_L_video_qa_msvd_l0_1_qsformerl10_llamav2vicuna_timesformer": "MSRVTQA (ViT)",
    "ePALM_L_audio_caption_l0_1_qsformerl10_llamav2vicuna_ast": "Audiocaps (ViT)",
    "ePALM_L_clotho_audio_caption_l0_1_qsformerl10_llamav2vicuna_ast": "Clotho (ViT)",
    "ePALM_L_clothoaqa_audio_qa_l0_1_qsformerl10_llamav2vicuna_ast": "Clotho-AQA (ViT)",
}


vicuna_folders_text = [
    "ePALM_caption_qformerl10_llamav2vicuna_vitl",
    "ePALM_L_video_caption_l0_1_qsformerl10_llamav2vicuna_timesformer",
    "ePALM_L_audio_caption_l0_1_qsformerl10_llamav2vicuna_ast",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        default="logs/DePalm/results/all_ious.json",
    )
    parser.add_argument("--mask_type", type=str, default="masks_onlypromptwithans")
    parser.add_argument("--mask_name", type=str, default="W_masks_s0.3.pth")
    parser.add_argument("--mode", type=str, default="prompt")
    parser.add_argument("--encoder", type=str, default="")
    parser.add_argument("--mask2_path", type=str, default="")
    parser.add_argument("--all_ious_dict", default=None)

    parser.add_argument("--text_mask", type=str, default="masks_onlytextwithans")
    parser.add_argument("--mask_dir", type=str, default="logs/DePalm")

    args = parser.parse_args()

    mask_type = args.mask_type
    mask_name = args.mask_name
    save_path = args.save_path
    mask2_path = args.mask2_path

    folders_ = folders_vicuna
    folder_2_name = vicuna_folder_2_name
    folders_text_ = vicuna_folders_text
    
    if "vicuna" in save_path:

        layer_names = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]  # llamav2

        if "with_text" not in args.mode:
            folders_text_ = None
        print(folders_, folders_text_, layer_names)
    else:
        layer_names = [
            "fc1",
            "fc2",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.out_proj",
        ]  # opt

        if "with_text" not in args.mode:
            folders_text_ = None

    if args.all_ious_dict is None:
        all_ious_dict = {k: {} for k in layer_names + ["avg"]}
    else:
        print(f"Reading results file from {args.all_ious_dict}")
        all_ious_dict = json.load(open(args.all_ious_dict))

    if args.mode == "text":
        all_ious_dict = get_ious_withtext(
            folders_,
            args.mask_dir,
            mask_type,
            mask_name,
            layer_names,
            all_ious_dict=all_ious_dict,
            startl=0,
            save_path=save_path,
        )
    else:
        all_ious_dict = get_ious(
            folders_,
            args.mask_dir,
            mask_type,
            mask_name,
            layer_names,
            all_ious_dict=all_ious_dict,
            startl=0,
            save_path=save_path,
            folders_text=folders_text_,
            mode=args.mode,
            mask2_path=mask2_path,
            text_mask=args.text_mask,
        )
