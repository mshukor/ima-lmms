## from VL-Adapter

import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from dataset.randaugment import RandomAugment
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

project_dir = Path(__file__).resolve().parent.parent  # VLT5
workspace_dir = project_dir.parent


class AOKVQAFineTuneDataset(Dataset):
    def __init__(
        self,
        split="train,valid",
        raw_dataset=None,
        rank=-1,
        topk=-1,
        verbose=True,
        mode="train",
        data_dir=None,
        image_size=224,
        use_data_augmentation=True,
    ):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose

        self.mode = mode

        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )

        self.image_size = image_size
        self.use_data_augmentation = use_data_augmentation

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.image_size, scale=(0.5, 1.0), interpolation=Image.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                RandomAugment(
                    2,
                    7,
                    isPIL=True,
                    augs=[
                        "Identity",
                        "AutoContrast",
                        "Equalize",
                        "Brightness",
                        "Sharpness",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                    ],
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size), interpolation=Image.BICUBIC
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

        ### dataset paths
        dataset_dir = Path(data_dir)

        self.sources = split.split(",")
        if self.verbose:
            print("Data sources: ", self.sources)

        self.img_ids_to_source = {}
        data_info_dicts = []
        for source in self.sources:
            data_info_path = dataset_dir.joinpath(f"{source}.json")
            with open(data_info_path) as f:
                _data_info_dicts = json.load(f)
                for _d in _data_info_dicts:
                    self.img_ids_to_source[_d["image_id"]] = source
                    _d["source"] = source

                data_info_dicts.extend(_data_info_dicts)
            if self.verbose:
                print(f"Loaded {len(_data_info_dicts)} data from", source)

        data = data_info_dicts

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

        if mode == "train" and self.use_data_augmentation:
            self.transform = self.train_transform
        else:
            self.transform = self.test_transform

        self.featname_to_h5 = {
            "train": dataset_dir.joinpath("train2017"),
            "val": dataset_dir.joinpath("val2017"),
            "test": dataset_dir.joinpath("val2017"),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}

        datum = self.data[idx]

        img_id = datum["image_id"]
        out_dict["img_id"] = img_id

        source = self.img_ids_to_source[img_id]
        path = self.featname_to_h5[source]

        path = path.joinpath(f"{img_id:012}.jpg")

        try:
            image = Image.open(path).convert("RGB")
        except FileNotFoundError:
            print(f"image not found {path}")
            new_idx = random.choice(list(range(len(self))))
            return self.__getitem__(idx)

        out_dict["image"] = self.transform(image)

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

        batch_entry["task"] = "AOKVQA"

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
    image_size=224,
    use_data_augmentation=True,
    **kwargs,
):

    _dset = AOKVQADataset(split, verbose, data_dir=data_dir)

    dataset = AOKVQAFineTuneDataset(
        split,
        raw_dataset=_dset,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        mode=mode,
        data_dir=data_dir,
        image_size=image_size,
        use_data_augmentation=use_data_augmentation,
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

    loader.evaluator = AOKVQAEvaluator(_dset)
    loader.task = "vqa"

    return loader, dataset


class AOKVQADataset:
    """
    A AOKVQA data example in json file:
    {
        "img_id": "2375429",
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

        dataset_dir = Path(data_dir)

        okvqa_dir = dataset_dir

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            data = json.load(open(okvqa_dir.joinpath("%s.json" % split)))
            self.data.extend(data)
        if verbose:
            print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {datum["question_id"]: datum for datum in self.data}

        # # Answers
        # self.ans2label = json.load(open(gqa_dir.joinpath("trainval_ans2label.json")))
        # self.label2ans = json.load(open(gqa_dir.joinpath("trainval_label2ans.json")))
        # assert len(self.ans2label) == len(self.label2ans)
        # for ans, label in self.ans2label.items():
        #     assert self.label2ans[label] == ans

    # @property
    # def num_answers(self):
    #     return len(self.data)

    def __len__(self):
        return len(self.data)


class AOKVQAEvaluator:
    def __init__(self, dataset: AOKVQADataset):
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

    def evaluate(self, quesid2ans: dict, normalize_answer=True):
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
                # print(ans, new_label)
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
        Dump the result to a AOKVQA-challenge submittable json file.
        AOKVQA json file submission requirement:
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
