import pandas as pd

import argparse
import os

file_names_from_datasets = [
    f"wandasparsitywithans{dataset}"
    for dataset in [
        "msvd",
        "msrvtqa",
        "msrvtt",
        "vqav2",
        "coco",
        "gqa",
        "okvqa",
        "audiocaps",
        "clotho",
        "clotho_aqa",
    ]
]
file_names_from_masks = ["maskfrom_ePALM"]
file_names_from_modality_masks = ["maskfrom_name"]
file_names_from_ex_modality_masks = ["maskfrom_ex_name_s0.015"]

file_names_from_intermag_modality_masks = ["maskfrom_intermag_name"]
file_names_from_uninonmag_modality_masks = ["maskfrom_uninonmag_name"]
file_names_from_uniontext_modality_masks = ["maskfrom_uniontext_name"]
file_names_from_intertext_modality_masks = ["maskfrom_intertext_name"]


def walk_dirs(root_dir):
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            yield entry.path
            yield from walk_dirs(entry.path)


def path_to_dataset(dir_name):
    if "msrvtqa" in dir_name:
        dataset = "MSRVTTQA"
    elif "msvd" in dir_name:
        dataset = "MSVD"
    elif "video_caption" in dir_name or "msrvtt" in dir_name:
        dataset = "MSRVTT"

    elif "clotho_aqa" in dir_name or "clothoaqa" in dir_name:
        dataset = "ClothoAQA"
    elif "clotho" in dir_name:
        dataset = "Clotho"
    elif "audio_caption" in dir_name or "audiocaps" in dir_name:
        dataset = "Audiocaps"

    elif "okvqa" in dir_name:
        dataset = "OKVQA"
    elif "gqa" in dir_name:
        dataset = "GQA"
    elif "vqa" in dir_name:
        dataset = "VQAv2"
    elif "caption" in dir_name or "coco" in dir_name:
        dataset = "COCO"

    else:
        dataset = None

    if "onlytext" in dir_name:
        dataset = dataset + " Text"

    return dataset


def path_to_modality(dir_name):
    if "audio" in dir_name:
        dataset = "audio"
    elif "video" in dir_name:
        dataset = "video"
    elif "image" in dir_name:
        dataset = "image"
    elif "text" in dir_name:
        dataset = "text"
    elif "random" in dir_name:
        dataset = "random"
    else:
        dataset = "all"

    return dataset


def get_score_name(dir_name):

    if any([k == dir_name for k in ["MSVD", "MSRVTTQA", "OKVQA", "VQAv2"]]):
        key = "valid_Valid/overall"
    elif any([k == dir_name for k in ["COCO", "MSRVTT", "Audiocaps", "Clotho"]]):
        key = "valid_Valid/CIDEr"
    elif any([k == dir_name for k in ["GQA", "ClothoAQA"]]):
        key = "valid_Valid/topk_score"
    else:
        raise NotImplemented

    return key


def get_encoder_name(dir_name):
    if any(
        [
            k in dir_name
            for k in [
                "clip",
                "clap",
            ]
        ]
    ):
        key = "CLIP"
    elif any([k in dir_name for k in ["vit", "timesformer", "ast"]]):
        key = "ViT"
    elif any([k in dir_name for k in ["mae"]]):
        key = "MAE"
    else:
        key = None

    return key


def are_last_three_decreasing(numbers):
    # Check if the last three numbers are in decreasing order
    for i in range(len(numbers) - 1):
        if numbers[i] < numbers[i + 1]:
            return False
    return True


columns = [
    "COCO",
    "VQAv2",
    "GQA",
    "OKVQA",
    "MSRVTT",
    "MSRVTTQA",
    "MSVD",
    "Audiocaps",
    "Clotho",
    "ClothoAQA",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./configs/VQA.yaml")
    parser.add_argument("--output_dir", type=str, default=0.5)
    parser.add_argument("--source", type=str, default="dataset")
    parser.add_argument("--source_mask", type=str, default="clip")
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--get_sparsity", action="store_true")
    parser.add_argument("--text_model", type=str, default=None)
    parser.add_argument("--num_sparsities", type=int, default=3)

    args = parser.parse_args()

    rows = columns.copy()

    if "ex_modality_mask" in args.source:
        rows = ["image", "video", "audio", "text", "random"]
    elif "modality_mask" in args.source:
        rows = ["image", "video", "audio", "text", "all"]
    elif args.source == "mask":
        columns += [x + " Text" for x in columns]
        rows = columns

    # Create an empty DataFrame to store the results

    encoder_to_df = {
        "CLIP": {
            "result_dfs0_3": pd.DataFrame(index=rows, columns=columns),
            "result_dfs0_5": pd.DataFrame(index=rows, columns=columns),
            "result_dfs0_7": pd.DataFrame(index=rows, columns=columns),
        },
        "MAE": {
            "result_dfs0_3": pd.DataFrame(index=rows, columns=columns),
            "result_dfs0_5": pd.DataFrame(index=rows, columns=columns),
            "result_dfs0_7": pd.DataFrame(index=rows, columns=columns),
        },
        "ViT": {
            "result_dfs0_3": pd.DataFrame(index=rows, columns=columns),
            "result_dfs0_5": pd.DataFrame(index=rows, columns=columns),
            "result_dfs0_7": pd.DataFrame(index=rows, columns=columns),
        },
    }

    if args.source == "dataset":
        file_names = file_names_from_datasets
    elif "mask" in args.source:
        if "ex_modality" in args.source:
            file_names = file_names_from_ex_modality_masks

        elif "intermag" in args.source:
            file_names = file_names_from_intermag_modality_masks
        elif "uninonmag" in args.source:
            file_names = file_names_from_uninonmag_modality_masks
        elif "uniontext" in args.source:
            file_names = file_names_from_uniontext_modality_masks
        elif "intertext" in args.source:
            file_names = file_names_from_intertext_modality_masks
        elif "modality" in args.source:
            file_names = file_names_from_modality_masks
        else:
            file_names = file_names_from_masks
    else:
        file_names = file_names_from_masks

    print(file_names)
    for folder in walk_dirs(args.results_dir):

        for file in os.listdir(folder):
            if ".csv" in file and any([m in file for m in file_names]):

                result_path = os.path.join(folder, file)

                if args.text_model is not None and args.text_model not in folder:
                    continue
                target = path_to_dataset(folder)
                if "modality" in args.source:
                    source = path_to_modality(file)
                elif "onlypromptwithans" in args.mode:
                    source = target
                else:
                    source = path_to_dataset(file)

                if target is None or source is None:
                    continue

                score_name = get_score_name(target)
                if args.get_sparsity:
                    score_name = "valid_sparsity"
                encoder_name = get_encoder_name(folder)

                if "mask" in args.source and "all" not in args.source_mask:
                    source_encoder_name = get_encoder_name(file)
                    if args.source_mask != source_encoder_name:
                        continue

                if any(
                    [k in args.source for k in ["modality", "intermag", "intertext"]]
                ):
                    if args.mode not in file:
                        continue

                if encoder_name is None:
                    print(f"error at {result_path}")
                    continue

                result_df = encoder_to_df[encoder_name]

                # Read the CSV file into a DataFrame
                csv_data = pd.read_csv(result_path, header=0)  # first line as header

                scores = list(csv_data[score_name])
                if "CIDEr" in score_name or any(
                    [k in target for k in ["GQA", "MSVD", "MRVTTQA", "ClothoAQA"]]
                ):
                    scores = [s * 100 for s in scores]

                scores = [round(s, 3) for s in scores]

                if len(scores) < args.num_sparsities and "modality" not in args.source:
                    continue

                if len(scores) > args.num_sparsities:
                    scores = scores[-args.num_sparsities :]
                    if not are_last_three_decreasing(scores):
                        continue

                if not any(
                    [k in args.source for k in ["modality", "intermag", "intertext"]]
                ):

                    sparsity_keys = ["result_dfs0_3", "result_dfs0_5", "result_dfs0_7"]
                    for i in range(args.num_sparsities):
                        result_df[sparsity_keys[i]].at[source, target] = scores[i]
                else:
                    if "0_3" in file:
                        sparsity = "0_3"
                    elif "0_5" in file:
                        sparsity = "0_5"
                    elif "0_7" in file:
                        sparsity = "0_7"

                    result_df[f"result_dfs{sparsity}"].at[source, target] = scores[-1]

                print(f"processed file: {result_path}")

    for k, v in encoder_to_df.items():
        encoder_to_df[k] = pd.concat(
            [
                encoder_to_df[k]["result_dfs0_3"],
                encoder_to_df[k]["result_dfs0_5"],
                encoder_to_df[k]["result_dfs0_7"],
            ]
        )
    result_df = pd.concat(
        [encoder_to_df["CLIP"], encoder_to_df["ViT"], encoder_to_df["MAE"]], axis=1
    )

    if args.text_model is not None and "llamav2" in args.text_model:
        print(encoder_to_df["ViT"])
    else:
        print(result_df)

    result_df.to_excel(args.output_dir)  # .xlsx
    print(f"saved at {args.output_dir}")
