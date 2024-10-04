import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class AudioCaptionFineTuneDataset(Dataset):
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
        melbins=128,
        target_length=1024,
        num_tries=1,
        freqm_p=24,
        timem_p=96,
        skip_norm=False,
        norm_mean=-4.2677393,
        norm_std=4.5689974,
        noise=False,
        processor=None,
    ):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        data_dir = Path(data_dir)
        dataset_dir = data_dir.joinpath("annotation")
        coco_img_dir = data_dir  # .joinpath('audios')

        self.black_image = black_image

        self.source = split
        if self.verbose:
            print("Data source: ", self.source)

        self.processor = processor
        # audio
        self.melbins = melbins
        self.target_length = target_length
        self.num_tries = num_tries  # 2
        self.freqm_p = freqm_p
        self.timem_p = timem_p
        self.skip_norm = skip_norm
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.noise = noise

        self.freqm = torchaudio.transforms.FrequencyMasking(self.freqm_p)
        self.timem = torchaudio.transforms.TimeMasking(self.timem_p)

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

                        img_id = ".".join(datum["audio"].split(".")[:-1])
                        new_datum = {
                            "img_id": img_id,
                            "sent": d.strip(),
                            "targets": [k.strip() for k in caption],
                            "is_train": True,
                            "audio": datum["audio"],
                        }
                        data.append(new_datum)
                else:
                    img_id = ".".join(datum["audio"].split(".")[:-1])
                    new_datum = {
                        "img_id": img_id,
                        "sent": caption.strip(),
                        "targets": caption.strip(),
                        "is_train": True,
                        "audio": datum["audio"],
                    }
                    data.append(new_datum)
            else:
                caption = datum["caption"]
                if not isinstance(caption, list):
                    caption = [caption]
                img_id = ".".join(datum["audio"].split(".")[:-1])
                new_datum = {
                    "img_id": img_id,
                    "targets": [d.strip() for d in caption],
                    "is_train": False,
                    "audio": datum["audio"],
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

                audio = datum["audio"]
                path = str(self.source_to_h5["all"].joinpath(f"{audio}"))

                waveform, sr = torchaudio.load(path)

                # if self.processor is None:
                waveform = waveform - waveform.mean()
                # audio
                fbank = torchaudio.compliance.kaldi.fbank(
                    waveform,
                    htk_compat=True,
                    sample_frequency=sr,
                    use_energy=False,
                    window_type="hanning",
                    num_mel_bins=self.melbins,
                    dither=0.0,
                    frame_shift=10,
                )
                # else:
                #     if waveform.ndim > 1:
                #         waveform = waveform[0]
                #     fbank = self.processor(waveform, sampling_rate=48000)['input_features'][0]
                #     fbank =  torch.tensor(fbank)

            except Exception as e:
                print(i, path)
                idx = random.randint(0, len(self) - 1)
                print(
                    f"Caught exception {e} when loading audio {path}, "
                    f"randomly sample a new audio as replacement"
                )
                continue

        # if self.processor is None:
        n_frames = fbank.shape[0]

        p = self.target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0 : self.target_length, :]

        # SpecAug, not do for eval set

        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
        fbank = fbank.unsqueeze(0)

        if self.mode == "train":
            if self.freqm_p != 0:
                fbank = self.freqm(fbank)
            if self.timem_p != 0:
                fbank = self.timem(fbank)
        # squeeze it back, it is just a trick to satisfy new torchaudio version
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.mode == "train" and self.noise == True:
            fbank = (
                fbank
                + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            )
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        out_dict["image"] = fbank

        if self.black_image:
            out_dict["image"] = torch.zeros_like(out_dict["image"])

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
    melbins=128,
    target_length=1024,
    num_tries=1,
    freqm_p=24,
    timem_p=96,
    skip_norm=False,
    norm_mean=-4.2677393,
    norm_std=4.5689974,
    noise=False,
    processor=None,
    metrics=None,
):

    dataset = AudioCaptionFineTuneDataset(
        split,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        mode=mode,
        data_dir=data_dir,
        black_image=black_image,
        melbins=melbins,
        target_length=target_length,
        num_tries=num_tries,
        freqm_p=freqm_p,
        timem_p=timem_p,
        skip_norm=skip_norm,
        norm_mean=norm_mean,
        norm_std=norm_std,
        noise=noise,
        processor=processor,
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
