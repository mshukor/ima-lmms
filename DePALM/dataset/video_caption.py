import json
import random
import re
from pathlib import Path

import torch
from dataset.video_utils import VIDEO_READER_FUNCS
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


class MSRVTTCaptionFineTuneDataset(Dataset):
    def __init__(
        self,
        split="karpathy_train",
        raw_dataset=None,
        rank=-1,
        topk=-1,
        verbose=True,
        args=None,
        mode="train",
        data_dir="/data/mshukor/data",
        black_image=False,
        num_frames=4,
        as_images=True,
        num_tries=1,
        sample_type="rand",
        image_size=224,
        use_data_augmentation=True,
    ):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        data_dir = Path(data_dir)
        dataset_dir = data_dir.joinpath("annotation")
        coco_img_dir = data_dir.joinpath("videos/all")

        self.black_image = black_image

        self.source = split
        if self.verbose:
            print("Data source: ", self.source)

        # video
        self.num_frames = num_frames  # 4
        self.max_num_frames = num_frames
        self.video_reader = VIDEO_READER_FUNCS["decord"]
        self.as_images = as_images  # True
        self.num_tries = num_tries  # 2
        self.sample_type = sample_type  # 'rand'
        self.image_size = image_size
        self.use_data_augmentation = use_data_augmentation

        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )

        type_transform = transforms.Lambda(lambda x: x.float().div(255.0))

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.5, 1.0), interpolation=Image.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(),
                type_transform,
                normalize,
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=Image.BICUBIC
                ),
                type_transform,
                normalize,
            ]
        )

        data_info_path = dataset_dir.joinpath(split + ".json")
        with open(data_info_path) as f:
            karpathy_data = json.load(f)

        n_images = 0

        data = []
        for datum in karpathy_data:

            if "train" in split:
                caption = datum["caption"]
                if isinstance(caption, list):
                    for d in caption:

                        img_id = ".".join(datum["video"].split(".")[:-1])
                        new_datum = {
                            "img_id": img_id,
                            "sent": d.strip(),
                            "targets": [k.strip() for k in caption],
                            "is_train": True,
                            "video": datum["video"],
                        }
                        data.append(new_datum)
                else:
                    img_id = ".".join(datum["video"].split(".")[:-1])
                    new_datum = {
                        "img_id": img_id,
                        "sent": caption.strip(),
                        "targets": caption.strip(),
                        "is_train": True,
                        "video": datum["video"],
                    }
                    data.append(new_datum)
            else:
                caption = datum["caption"]
                if not isinstance(caption, list):
                    caption = [caption]
                img_id = ".".join(datum["video"].split(".")[:-1])
                new_datum = {
                    "img_id": img_id,
                    "targets": [d.strip() for d in caption],
                    "is_train": False,
                    "video": datum["video"],
                }
                data.append(new_datum)

            n_images += 1

        if self.verbose:
            print(f"{self.source} has {n_images} images")
            print(f"Loaded {len(data)} data from", split)

        if isinstance(self.topk, float) and (0 < self.topk <= 1):
            used_samples = int(self.topk * len(data))
            data = random.sample(data, used_samples)
            if self.verbose:
                print(f"Use only {len(data)} data")

        elif self.topk > 0:
            data = data[: int(self.topk)]
            if self.verbose:
                print(f"Use only {len(data)} data")

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))

        self.image_size = self.image_size

        if mode == "train" and self.use_data_augmentation:
            self.transform = self.train_transform
        else:
            self.transform = self.test_transform

        self.source_to_h5 = {}

        self.source_to_h5.update(
            {
                "all": coco_img_dir,
            }
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict["args"] = self.args

        for i in range(self.num_tries):

            try:
                datum = self.data[idx]

                ###### Image ######
                img_id = datum["img_id"]
                out_dict["img_id"] = img_id

                video = datum["video"]
                path = str(self.source_to_h5["all"].joinpath(f"{video}"))

                max_num_frames = (
                    self.max_num_frames if hasattr(self, "max_num_frames") else -1
                )
                frames, frame_indices, video_duration = self.video_reader(
                    path,
                    self.num_frames,
                    self.sample_type,
                    max_num_frames=max_num_frames,
                )
                if frames.shape[0] < self.num_frames:
                    frames, frame_indices, video_duration = self.video_reader(
                        path, self.num_frames, "rand", max_num_frames=max_num_frames
                    )

            except Exception as e:
                print(i, path)
                idx = random.randint(0, len(self) - 1)
                print(
                    f"Caught exception {e} when loading video {path}, "
                    f"randomly sample a new video as replacement"
                )
                continue

        out_dict["image"] = self.transform(frames)

        if self.black_image:
            out_dict["image"] = torch.zeros_like(out_dict["image"])

        if not self.as_images:
            out_dict["image"] = out_dict["image"].permute(1, 0, 2, 3)  # -> CTHW

        if datum["is_train"]:
            sent = datum["sent"].strip()

            out_dict["sent"] = sent

        if "targets" in datum:
            out_dict["targets"] = datum["targets"]

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        if "target_ids" in batch[0]:
            T_W_L = max(entry["target_length"] for entry in batch)
            target_ids = (
                torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
            )

        targets = []
        img_ids = []
        img_paths = []
        input_text = []
        images = []
        sents = []

        for i, entry in enumerate(batch):

            images.append(entry["image"])
            img_ids.append(entry["img_id"])

            if "target_ids" in entry:
                target_ids[i, : entry["target_length"]] = entry["target_ids"]

            if "targets" in entry:
                targets.append(entry["targets"])
            if "sent" in entry:
                sents.append(entry["sent"])

        # if self.args.use_vision:
        batch_entry["images"] = torch.stack(images)
        batch_entry["img_id"] = img_ids
        batch_entry["img_paths"] = img_paths
        if "sent" in entry:
            batch_entry["sent"] = sents

        batch_entry["targets"] = targets

        batch_entry["task"] = "caption"

        return batch_entry


def pre_caption(caption, max_words):
    caption = (
        re.sub(
            r"([,.'!?\"()*#:;~])",
            "",
            caption.lower(),
        )
        .replace("-", " ")
        .replace("/", " ")
        .replace("<person>", "person")
    )

    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    # truncate caption
    caption_words = caption.split(" ")
    if len(caption_words) > max_words:
        caption = " ".join(caption_words[:max_words])

    return caption


def get_loader(
    split="train",
    mode="train",
    batch_size=32,
    workers=4,
    distributed=False,
    gpu=0,
    topk=-1,
    data_dir="/data/mshukor/data",
    local_rank=None,
    world_size=None,
    verbose=False,
    config_dir=None,
    black_image=False,
    num_frames=4,
    as_images=True,
    num_tries=1,
    sample_type="rand",
    image_size=224,
    use_data_augmentation=True,
    metrics=None,
    **kwargs,
):

    dataset = MSRVTTCaptionFineTuneDataset(
        split,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        mode=mode,
        data_dir=data_dir,
        black_image=black_image,
        num_frames=num_frames,
        as_images=as_images,
        num_tries=num_tries,
        sample_type=sample_type,
        image_size=image_size,
        use_data_augmentation=use_data_augmentation,
    )

    if distributed and mode == "train":
        train_sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=local_rank
        )
    else:
        train_sampler = None
    if mode == "train":
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=workers,
            pin_memory=True,
            sampler=train_sampler,
            collate_fn=dataset.collate_fn,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            sampler=None,
            collate_fn=dataset.collate_fn,
            drop_last=False,
        )

    if verbose:
        loader.evaluator = COCOCaptionEvaluator(metrics=metrics)

    loader.task = "caption"

    return loader, dataset


class COCOCaptionEvaluator:
    def __init__(self, metrics=None):
        import language_evaluation

        if metrics is not None:
            self.evaluator = language_evaluation.CocoEvaluator(
                verbose=False, coco_types=metrics
            )  # coco_types=["BLEU", "METEOR", "ROUGE_L", "CIDEr", "SPICE"]
        else:
            self.evaluator = language_evaluation.CocoEvaluator(
                verbose=False
            )  # coco_types=["BLEU", "METEOR", "ROUGE_L", "CIDEr", "SPICE"]

    def evaluate(self, predicts, answers):

        results = self.evaluator.run_evaluation(predicts, answers)

        return results
