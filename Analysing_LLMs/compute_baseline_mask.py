import numpy as np
import torch
from tqdm import tqdm

import argparse
import os
from functools import reduce


main_mask_dir = "logs/DePalm"
mask_dirs = ["masks_onlypromptwithans", "masks_withans", "masks_onlytextwithans"]




#### Vicuna
all_llamav2vicuna_image_masks = [
    "ePALM_caption_qformerl10_llamav2vicuna_vitl",
    "ePALM_vqa_qformerl10_llamav2vicuna_vitl",
    "ePALM_gqa_qformerl10_llamav2vicuna_vitl",
    "ePALM_okvqa_qformerl10_llamav2vicuna_vitl",
]

########## videos
all_llamav2vicuna_video_masks = [
    "ePALM_L_video_caption_l0_1_qsformerl10_llamav2vicuna_timesformer",
    "ePALM_L_video_qa_msvd_l0_1_qsformerl10_llamav2vicuna_timesformer",
]

########## audios
all_llamav2vicuna_audio_masks = [
    "ePALM_L_audio_caption_l0_1_qsformerl10_llamav2vicuna_ast",
    "ePALM_L_clothoaqa_audio_qa_l0_1_qsformerl10_llamav2vicuna_ast",
]


all_llamav2vicuna_masks = [
    "ePALM_caption_qformerl10_llamav2vicuna_vitl",
    "ePALM_vqa_qformerl10_llamav2vicuna_vitl",
    "ePALM_gqa_qformerl10_llamav2vicuna_vitl",
    "ePALM_okvqa_qformerl10_llamav2vicuna_vitl",
    "ePALM_L_video_caption_l0_1_qsformerl10_llamav2vicuna_timesformer",
    "ePALM_L_video_qa_msvd_l0_1_qsformerl10_llamav2vicuna_timesformer",
    "ePALM_L_audio_caption_l0_1_qsformerl10_llamav2vicuna_ast",
    "ePALM_L_clothoaqa_audio_qa_l0_1_qsformerl10_llamav2vicuna_ast",
]

all_llamav2vicuna_masks_from_path_s0_5 = [
    "logs/DePalm/masks/masks_onlytextwithans_all_llamav2vicuna_masks_W_masks_s0_5.pth",
    "logs/DePalm/masks/masks_onlypromptwithans_all_llamav2vicuna_masks_W_masks_s0_5.pth",
]

all_llamav2vicuna_masks_from_path_s0_3 = [
    "logs/DePalm/masks/masks_onlytextwithans_all_llamav2vicuna_masks_W_masks_s0_3.pth",
    "logs/DePalm/masks/masks_onlypromptwithans_all_llamav2vicuna_masks_W_masks_s0_3.pth",
]



def print_sparsity(mask_1):
    sub_count = 0
    sub_params = 0
    for i in range(len(mask_1)):
        for layer_name in layer_names:
            sub_count += (mask_1[i][layer_name] == 1).sum().item()
            sub_params += mask_1[i][layer_name].numel()
    print(f"sparsity {float(sub_count)/sub_params:.6f}")





def compute_ex_mask(mask_1, mask_2, layer_names=["fc2"], startl=0):
    mask = {i: {} for i in range(len(mask_1))}
    for i in tqdm(range(len(mask_1))):
        if i >= startl:
            sub_count = 0
            sub_params = 0
            for layer_name in layer_names:
                matrix1 = ~mask_1[i][
                    layer_name
                ].numpy()  # so 1 for valid and 0 for masked weight
                matrix2 = ~mask_2[i][layer_name].numpy()

                # intersection_matrix = matrix1 & matrix2
                ex_matrix_1 = matrix1 & (~matrix2)
                # ex_matrix_1 = matrix1 & (matrix2)

                mask[i][layer_name] = torch.tensor(~ex_matrix_1)

                sub_count += (mask[i][layer_name] == 1).sum().item()
                sub_params += mask[i][layer_name].numel()
            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

        else:
            print(f"skip {i} {layer_name}")
    return mask


def add_mask(mask_1, mask_2, layer_names=["fc2"], startl=0):
    mask = {i: {} for i in range(len(mask_1))}
    for i in tqdm(range(len(mask_1))):
        if i >= startl:
            for layer_name in layer_names:
                matrix1 = mask_1[i][layer_name].numpy()
                matrix2 = mask_2[i][layer_name].numpy()

                ex_matrix_1 = np.logical_or(matrix1, matrix2)

                mask[i][layer_name] = torch.tensor(ex_matrix_1)

        else:
            print(f"skip {i} {layer_name}")
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_list", default="./configs/VQA.yaml")
    parser.add_argument("--output_dir", type=str, default=0.5)
    parser.add_argument("--mask_dir", type=str, default=0.5)
    parser.add_argument("--mask_name", type=str, default=0.5)
    parser.add_argument("--mode", type=str, default="intersection")  # or avg?
    parser.add_argument("--main_mask_dir", type=str, default="logs/DePalm")  # or avg?

    args = parser.parse_args()

    mask_list = globals().get(args.mask_list)

    masks = []
    print(f"mask_list: {mask_list}")

    for mask in tqdm(mask_list):
        if "path" not in args.mask_list:
            mask_dir_path = os.path.join(args.main_mask_dir, mask)
            mask_path = os.path.join(mask_dir_path, args.mask_dir, args.mask_name)
        else:
            mask_path = mask

        if os.path.exists(mask_path):
            try:
                masks.append(torch.load(mask_path))
                print(f"loading mask at {mask_path}")
            except:
                print(f"skip mask at {mask_path}, error")
        else:
            print(f"skip mask at {mask_path}, not found")

    nb_layers = len(masks[0])

    layer_names = masks[0][0].keys()
    print(layer_names)
    if args.mode == "complement":
        print_sparsity(masks[0])
        print_sparsity(masks[1])
        MASK = compute_ex_mask(masks[0], masks[1], layer_names=layer_names)
    elif args.mode == "add":
        MASK = add_mask(masks[0], masks[1], layer_names=layer_names)
    else:

        MASK = {i: {} for i in range(nb_layers)}

        for i in tqdm(range(nb_layers)):
            sub_count = 0
            sub_params = 0

            for layer_name in layer_names:

                W_masks = []
                for j in range(0, len(masks)):
                    if args.mode == "avg":
                        W_masks.append(float(~masks[j][i][layer_name]))
                    else:
                        W_masks.append(~masks[j][i][layer_name])

                if args.mode == "avg":  # to be multiplied with the original weights
                    W_masks = torch.stack(W_masks, dim=0)
                    W_masks = torch.mean(W_masks, dim=0)
                else:
                    W_masks = reduce(torch.logical_and, W_masks)

                MASK[i][layer_name] = ~W_masks

                sub_count += (MASK[i][layer_name] == 1).sum().item()
                sub_params += MASK[i][layer_name].numel()
            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    if "path" not in args.mask_list:
        save_path = os.path.join(
            args.output_dir,
            f"{args.mask_dir}_{args.mask_list}_{'_'.join(args.mask_name.split('.')[:2])}.pth",
        )
    else:
        save_path = os.path.join(args.output_dir, f"{args.mask_list}.pth")

    os.makedirs(args.output_dir, exist_ok=True)

    torch.save(MASK, save_path)
    print(f"saved at {save_path}")
