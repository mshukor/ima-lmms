import json
import random
import re

import torch
from dataset.randaugment import RandomAugment
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def pre_question(question, max_ques_words):
    question = (
        re.sub(
            r"([,.'!?\"()*#:;~])",
            "",
            question.lower(),
        )
        .replace("-", " ")
        .replace("/", " ")
    )
    question = question.rstrip(" ")

    # truncate question
    question_words = question.split(" ")
    if len(question_words) > max_ques_words:
        question = " ".join(question_words[:max_ques_words])

    return question


###############################################
# https://raw.githubusercontent.com/ylsung/VL_adapter/545fcbbdbbaec4c442de35567f6ae477ff4e8265/VL-T5/src/vqa_raw_data.py


import json
import random
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


class VQAFineTuneDataset(Dataset):
    def __init__(
        self,
        split="train",
        raw_dataset=None,
        rank=-1,
        topk=-1,
        verbose=True,
        mode="train",
        data_dir=None,
        image_size=224,
        use_data_augmentation=True,
        raw_label=False,
    ):
        super().__init__()

        dataset_dir = Path(data_dir)

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.raw_label = raw_label
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

        # Loading datasets to data
        self.sources = split.split(",")
        if self.verbose:
            print("Data sources: ", self.sources)

        self.answer_normalizer = VQAEvaluator()

        data = raw_dataset.data

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

        self.source_to_h5 = {
            "train2014": dataset_dir.joinpath(f"train2014"),
            "val2014": dataset_dir.joinpath(f"val2014"),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}

        datum = self.data[idx]

        ###### Image ######
        # print(datum)
        img_id = datum["image_id"]
        img = datum["image"]
        out_dict["img_id"] = img_id

        out_dict["img"] = img

        path = self.source_to_h5[datum["split"]].joinpath(f"{img}.jpg")

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

        if "is_topk_optimal" in datum:
            out_dict["is_topk_optimal"] = datum["is_topk_optimal"]

        if "answers" in datum:

            if "label" not in datum:
                label = datum["answers"]
                count = {}
                for a in label:
                    ans = a["answer"]
                    if ans in count:
                        count[ans] += 1
                    else:
                        count[ans] = 1
                labels = {}
                for k, v in count.items():
                    labels[k] = min([v / 3, 1])

                out_dict["label"] = labels
            else:
                out_dict["label"] = datum["label"]

            answers = datum["answers"]
            answer = random.choice(answers)["answer"]

            score = int(len(answers) > 0)

            out_dict["answer"] = answer
            out_dict["score"] = score
            out_dict["all_answers"] = [a["answer"] for a in answers]

            # if self.raw_label or 'train' in self.mode:

            #     # 10 raw answers
            #     # ex) 'answers': [{'answer': 'net', 'answer_confidence': 'maybe', 'answer_id': 1},
            #     #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 2},
            #     #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 3},
            #     #     {'answer': 'netting', 'answer_confidence': 'yes', 'answer_id': 4},
            #     #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 5},
            #     #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 6},
            #     #     {'answer': 'mesh', 'answer_confidence': 'maybe', 'answer_id': 7},
            #     #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 8},
            #     #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 9},
            #     #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 10}],

            #     answers = datum['answers']
            #     answer = random.choice(answers)['answer']

            #     if self.answer_normalize:
            #         answer = self.answer_normalizer.normalize_answer(answer)

            #     score = int(len(answers) > 0)

            #     out_dict['answer'] = answer
            #     out_dict['score'] = score
            #     out_dict['all_answers'] = [a['answer'] for a in answers]

            # else:
            #     # https://github.com/airsplay/lxmert/blob/master/src/pretrain/lxmert_pretrain.py#L191

            #     answers = []
            #     scores = []
            #     for a, s in label.items():
            #         answers.append(a)
            #         scores.append(s)

            #     score_sum = sum(scores)

            #     if score_sum == 0:
            #         answer = ''
            #         score = 0.
            #     else:
            #         prob = [score / score_sum for score in scores]
            #         choice = np.random.multinomial(1, prob).argmax()
            #         answer = answers[choice]
            #         score = scores[choice]
            #         assert len(answer) > 0, (sent, label, choice, answer)

            #     out_dict['answer'] = answer
            #     out_dict['score'] = score
            #     out_dict['all_answers'] = answers

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        sentences = []
        question_ids = []
        answers = []
        all_answers = []
        img_ids = []
        img_paths = []
        labels = []
        scores = []
        is_topk_optimal = []
        images = []

        for i, entry in enumerate(batch):

            images.append(entry["image"])

            if "target" in entry:
                targets[i] += entry["target"]

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

            if "is_topk_optimal" in entry:
                is_topk_optimal.append(entry["is_topk_optimal"])

            if "img_id" in entry:
                img_ids.append(entry["img_id"])

        if "target" in batch[0]:
            batch_entry["targets"] = targets

        batch_entry["sent"] = sentences
        batch_entry["question_ids"] = question_ids
        batch_entry["answers"] = answers
        batch_entry["all_answers"] = all_answers
        batch_entry["scores"] = torch.FloatTensor(scores)
        batch_entry["labels"] = labels
        batch_entry["img_ids"] = img_ids

        batch_entry["task"] = "vqa"
        batch_entry["images"] = torch.stack(images)

        return batch_entry


def get_loader(
    split="train",
    mode="train",
    batch_size=32,
    workers=4,
    distributed=False,
    gpu=0,
    topk=-1,
    verbose=False,
    data_dir="/data/mshukor/data",
    local_rank=None,
    world_size=None,
    image_size=224,
    use_data_augmentation=True,
    **kwargs,
):

    # verbose = (gpu == 0)

    _dset = VQADataset(split, verbose, data_dir=data_dir)

    dataset = VQAFineTuneDataset(
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

    if verbose:
        loader.evaluator = VQAEvaluator(_dset)

    loader.task = "vqa"

    return loader, dataset


class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """

    def __init__(self, splits: str, verbose=True, data_dir="/data/mshukor/data"):
        self.name = splits
        self.splits = splits.split(",")

        dataset_dir = Path(data_dir)

        with open(dataset_dir.joinpath(f"mscoco_train2014_annotations.json")) as f:
            train2014_data = json.load(f)
        with open(
            dataset_dir.joinpath(f"OpenEnded_mscoco_train2014_questions.json")
        ) as f:
            train2014_questions = json.load(f)
        train_qid_to_q = {}
        for q in train2014_questions["questions"]:
            train_qid_to_q[q["question_id"]] = q["question"]
        for d in train2014_data["annotations"]:
            d["question"] = train_qid_to_q[d["question_id"]]
            d["split"] = "train2014"
            d["image"] = f"COCO_train2014_{d['image_id']:012}"  # 000000223914'

        with open(dataset_dir.joinpath(f"mscoco_val2014_annotations.json")) as f:
            val2014_data = json.load(f)
        with open(
            dataset_dir.joinpath(f"OpenEnded_mscoco_val2014_questions.json")
        ) as f:
            val2014_questions = json.load(f)
        val_qid_to_q = {}
        for q in val2014_questions["questions"]:
            val_qid_to_q[q["question_id"]] = q["question"]
        for d in val2014_data["annotations"]:
            d["question"] = val_qid_to_q[d["question_id"]]
            d["split"] = "val2014"
            d["image"] = f"COCO_val2014_{d['image_id']:012}"  # 000000223914'

        train2014_id2datum = {}
        for datum in train2014_data["annotations"]:
            qid = datum["question_id"]
            train2014_id2datum[qid] = datum
        val2014_id2datum = {}
        for datum in val2014_data["annotations"]:
            qid = datum["question_id"]
            val2014_id2datum[qid] = datum
        self.id2datum_gt = {**train2014_id2datum, **val2014_id2datum}

        # Loading datasets
        self.data = []
        for split in self.splits:
            if split == "train":
                self.data.extend(train2014_data["annotations"])
            elif split == "val":
                self.data.extend(val2014_data["annotations"])

        if verbose:
            print("Load %d data from split(s) %s." % (len(self.data), self.name))

        ## create labels
        # self.data
        # Convert list to dict (for evaluation)
        if "answers" in self.data[0] and "label" not in self.data[0]:
            for datum in self.data:
                label = datum["answers"]
                count = {}
                for a in label:
                    ans = a["answer"]
                    if ans in count:
                        count[ans] += 1
                    else:
                        count[ans] = 1
                labels = {}
                for k, v in count.items():
                    labels[k] = min([v / 3, 1])

                datum["labels"] = labels

        self.id2datum = {datum["question_id"]: datum for datum in self.data}

        # # Topk Answers
        # self.ans2label = json.load(
        #     open(vqa_dir.joinpath("trainval_ans2label.json")))
        # self.label2ans = json.load(
        #     open(vqa_dir.joinpath("trainval_label2ans.json")))
        # assert len(self.ans2label) == len(self.label2ans)

        # if verbose:
        #     print('# Answers:', len(self.ans2label))

    # @property
    # def num_answers(self):
    #     return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class VQAEvaluator:
    def __init__(self, dataset: VQADataset = None):
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

        self.n = 2

    def evaluate(self, quesid2ans: dict):
        score = 0.0
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum["labels"]
            if ans in label:
                try:
                    score += label[ans]
                except TypeError:
                    print("ans", ans)
                    print("label", label)
                    print(datum)
                    raise
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }
        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, "w") as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({"question_id": ques_id, "answer": ans})
            json.dump(result, f, indent=4, sort_keys=True)

    def evaluate_raw(self, quesid2ans: dict, is_topk_optimal=None):
        """https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py"""

        gts = self.dataset.id2datum_gt

        self.accuracy = {}
        self.evalQA = {}
        self.evalQuesType = {}
        self.evalAnsType = {}

        accQA = []
        accQuesType = {}
        accAnsType = {}

        for quesId, resAns in tqdm(quesid2ans.items(), total=len(quesid2ans), ncols=80):

            quesId = int(quesId)

            datum = self.dataset.id2datum[quesId]

            if is_topk_optimal is None:
                pass
            elif "is_topk_optimal" in datum:
                if datum["is_topk_optimal"] != is_topk_optimal:
                    continue

            resAns = resAns.replace("\n", " ")
            resAns = resAns.replace("\t", " ")
            resAns = resAns.strip()
            resAns = self.processPunctuation(resAns)
            resAns = self.processDigitArticle(resAns)

            gtAcc = []
            gtAnswers = [ans["answer"] for ans in gts[quesId]["answers"]]
            if len(set(gtAnswers)) > 1:
                for ansDic in gts[quesId]["answers"]:
                    ans = self.processPunctuation(ansDic["answer"])
                    ans = self.processDigitArticle(ans)
                    ansDic["answer"] = ans

            for gtAnsDatum in gts[quesId]["answers"]:
                otherGTAns = [
                    item for item in gts[quesId]["answers"] if item != gtAnsDatum
                ]
                matchingAns = [item for item in otherGTAns if item["answer"] == resAns]
                acc = min(1, float(len(matchingAns)) / 3)
                gtAcc.append(acc)
            quesType = gts[quesId]["question_type"]
            ansType = gts[quesId]["answer_type"]
            avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
            accQA.append(avgGTAcc)
            if quesType not in accQuesType:
                accQuesType[quesType] = []
            accQuesType[quesType].append(avgGTAcc)
            if ansType not in accAnsType:
                accAnsType[ansType] = []
            accAnsType[ansType].append(avgGTAcc)

            self.setEvalQA(quesId, avgGTAcc)
            self.setEvalQuesType(quesId, quesType, avgGTAcc)
            self.setEvalAnsType(quesId, ansType, avgGTAcc)

        if len(accQA) == 0:
            return {"overall": 0, "perQuestionType": {}, "perAnswerType": {}}
        else:
            self.setAccuracy(accQA, accQuesType, accAnsType)

        return self.accuracy

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

    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100 * acc, self.n)

    def setEvalQuesType(self, quesId, quesType, acc):
        if quesType not in self.evalQuesType:
            self.evalQuesType[quesType] = {}
        self.evalQuesType[quesType][quesId] = round(100 * acc, self.n)

    def setEvalAnsType(self, quesId, ansType, acc):
        if ansType not in self.evalAnsType:
            self.evalAnsType[ansType] = {}
        self.evalAnsType[ansType][quesId] = round(100 * acc, self.n)

    def setAccuracy(self, accQA, accQuesType, accAnsType):
        self.accuracy["overall"] = round(100 * float(sum(accQA)) / len(accQA), self.n)
        self.accuracy["perQuestionType"] = {
            quesType: round(
                100 * float(sum(accQuesType[quesType])) / len(accQuesType[quesType]),
                self.n,
            )
            for quesType in accQuesType
        }
        self.accuracy["perAnswerType"] = {
            ansType: round(
                100 * float(sum(accAnsType[ansType])) / len(accAnsType[ansType]), self.n
            )
            for ansType in accAnsType
        }
