## from VL-Adapter

import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

project_dir = Path(__file__).resolve().parent.parent  # VLT5
workspace_dir = project_dir.parent


class CLOTHOVQAFineTuneDataset(Dataset):
    def __init__(
        self,
        split="train,valid",
        raw_dataset=None,
        rank=-1,
        topk=-1,
        verbose=True,
        mode="train",
        data_dir=None,
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

        self.mode = mode

        data_dir = Path(data_dir)
        dataset_dir = data_dir.joinpath("annotation")
        coco_img_dir = data_dir

        # audio
        self.processor = processor
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

        data = karpathy_data

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

        datum = self.data[idx]

        for i in range(self.num_tries):

            try:
                datum = self.data[idx]

                ###### Image ######
                audio = datum["audio"]
                out_dict["img_id"] = audio.split(".")[0]

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

        ###### Text #####

        if "sent" in datum:
            sent = datum["sent"]
        elif "question" in datum:
            sent = datum["question"]

        question_id = datum["question_id"]
        out_dict["question_id"] = question_id

        out_dict["sent"] = sent

        if "label" in datum:
            label = datum["label"]
            out_dict["label"] = label

            # https://github.com/airsplay/lxmert/blob/master/src/pretrain/lxmert_pretrain.py#L191
            answers = []
            scores = []
            for a, s in label.items():
                answers.append(a)
                scores.append(s)

            score_sum = sum(scores)

            if score_sum == 0:
                answer = ""
                score = 0.0
            else:
                prob = [score / score_sum for score in scores]
                choice = np.random.multinomial(1, prob).argmax()
                answer = answers[choice]
                score = scores[choice]
                assert len(answer) > 0, (sent, label, choice, answer)

            out_dict["answer"] = answer
            out_dict["score"] = score
            out_dict["all_answers"] = answers

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        sentences = []
        question_ids = []
        answers = []
        all_answers = []
        labels = []
        scores = []

        images = []

        for i, entry in enumerate(batch):

            images.append(entry["image"])

            sentences.append(entry["sent"])
            question_ids.append(entry["question_id"])
            if "answer" in entry:
                answers.append(entry["answer"])
            if "all_answers" in entry:
                all_answers.append(entry["all_answers"])

            if "score" in entry:
                scores.append(entry["score"])

            if "label" in entry:
                labels.append(entry["label"])

        batch_entry["images"] = torch.stack(images)

        batch_entry["sent"] = sentences
        batch_entry["question_ids"] = question_ids
        batch_entry["answers"] = answers
        batch_entry["all_answers"] = all_answers

        batch_entry["scores"] = torch.FloatTensor(scores)
        batch_entry["labels"] = labels

        batch_entry["task"] = "gqa"

        return batch_entry


def get_loader(
    split="train",
    mode="train",
    batch_size=32,
    workers=4,
    distributed=False,
    gpu=0,
    topk=-1,
    verbose=None,
    data_dir="/data/mshukor/data",
    local_rank=None,
    world_size=None,
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
    **kwargs,
):

    _dset = CLOTHOVQADataset(split, verbose, data_dir=data_dir)

    dataset = CLOTHOVQAFineTuneDataset(
        split,
        raw_dataset=_dset,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        mode=mode,
        data_dir=data_dir,
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

    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
    else:
        sampler = None

    if mode == "train":
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            num_workers=workers,
            pin_memory=True,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False,
        )

    loader.evaluator = CLOTHOVQAQAEvaluator(_dset)
    loader.task = "vqa"

    return loader, dataset


class CLOTHOVQADataset:
    """
    A GQA data example in json file:
    {
        "video": "2375429.mp4",
        "label": {
            "pipe": 1.0
        },
        "question_id": "07333408",
        "sent": "What is on the white wall?"
    }
    """

    def __init__(self, splits: str, verbose=True, data_dir="/data/mshukor/data"):
        self.name = splits
        self.splits = splits.split(",")

        data_dir = Path(data_dir)
        dataset_dir = data_dir.joinpath("annotation")

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(dataset_dir.joinpath("%s.json" % split))))
        if verbose:
            print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {datum["question_id"]: datum for datum in self.data}

    def __len__(self):
        return len(self.data)


class CLOTHOVQAQAEvaluator:
    def __init__(self, dataset: CLOTHOVQADataset):
        self.dataset = dataset

        """https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py"""

        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }

        self.manualMap = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }

        self.articles = ["a", "an", "the"]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def evaluate(self, quesid2ans: dict, normalize_answer=False):
        score = 0.0
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum["label"]
            if normalize_answer:
                ans = self.normalize_answer(ans)
                new_label = {self.normalize_answer(l): label[l] for l in label}
            else:
                new_label = label

            if ans in new_label:
                score += new_label[ans]
        return score / len(quesid2ans)

    def normalize_answer(self, resAns):
        resAns = resAns.replace("\n", " ")
        resAns = resAns.replace("\t", " ")
        resAns = resAns.strip()
        resAns = self.processPunctuation(resAns)
        resAns = self.processDigitArticle(resAns)
        resAns = resAns.replace(",", "")
        return resAns

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump the result to a GQA-challenge submittable json file.
        GQA json file submission requirement:
            results = [result]
            result = {
                "questionId": str,      # Note: it's a actually an int number but the server requires an str.
                "prediction": str
            }
        :param quesid2ans: A dict mapping question id to its predicted answer.
        :param path: The file path to save the json file.
        :return:
        """
        with open(path, "w") as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({"questionId": ques_id, "prediction": ans})
            json.dump(result, f, indent=4, sort_keys=True)
