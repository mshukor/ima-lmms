import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import argparse
import datetime
import os
import random
import ruamel_yaml as yaml
import sys
import time
from pathlib import Path

sys.path.append(os.path.abspath("."))


import re
import time
import utils
from accelerate import Accelerator
from calflops.calculate_pipline import CalFlopsPipline
from calflops.utils import number_to_string
from dataset.aokvqa import get_loader as get_loader_aokvqa
from dataset.audio_caption import get_loader as get_loader_audiocaps
from dataset.audio_vqa import get_loader as get_loader_clothoaqa
from dataset.caption import get_loader
from dataset.gqa import get_loader as get_loader_gqa
from dataset.okvqa import get_loader as get_loader_okvqa
from dataset.textvqa import get_loader as get_loader_textvqa
from dataset.video_caption import get_loader as get_loader_msrvtt
from dataset.video_vqa import get_loader as get_loader_msrvttqa
from dataset.vqa import get_loader as get_loader_vqa
from functools import partial
from itertools import chain
from models.epalmv2 import ePALMv2, smart_tokenizer_and_embedding_resize
from models.utils import (
    exclude_list, filter_msg, filter_state, freeze_whole_model, print_trainable_params_percentage,
    remove_letters, save_logs, unfreeze_parameters)
from optim import create_optimizer
## pruning
from pruning.wanda_mm import check_sparsity, prune_given_mask, prune_magnitude, prune_wanda
from scheduler import create_scheduler
from transformers import AutoTokenizer, LlamaTokenizer

INSTRUCTIONS = [
    "Describe the image:",
]


def train(
    model,
    data_loader,
    optimizer,
    tokenizer,
    epoch,
    warmup_steps,
    device,
    scheduler,
    config,
    accelerator=None,
):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ", accelerator=accelerator)
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
    )

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size
    lm_loss_weight = config.get("lm_loss_weight", 1)
    append_eos_token = config.get("append_eos_token", False)
    eos_token = tokenizer.eos_token

    config_optim = utils.AttrDict(config["optimizer"])
    prompt_lr = config_optim.prompt_lr if hasattr(config_optim, "prompt_lr") else None

    instruction = config.get("instruction", "")
    accelerator.print(f"Use the following instructions: {instruction}")

    if prompt_lr is not None:
        metric_logger.add_meter(
            "prompt_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
        )

    task = data_loader.task

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        image = batch["images"].to(device, non_blocking=True)
        text = batch["sent"]

        if task == "vqa":
            answer = batch["answers"]
            text_ = [
                f"Question:{text[i]} Answer:{answer[i].replace('[SEP]','')}"
                for i in range(len(text))
            ]
            question = [f"Question:{text[i]} Answer:" for i in range(len(text))]
            if append_eos_token:
                text_ = [t.replace(eos_token, "") + eos_token for t in text_]
            text_input = tokenizer(text_, padding="longest", return_tensors="pt").to(
                device
            )
            question_input = tokenizer(
                question, padding="longest", return_tensors="pt"
            ).to(device)
        else:
            if isinstance(instruction, list):
                inst = random.choice(instruction)
            else:
                inst = instruction
            text = [inst + t for t in text]
            if append_eos_token:
                text = [t.replace(eos_token, "") + eos_token for t in text]
            # print(text)
            text_input = tokenizer(text, padding="longest", return_tensors="pt").to(
                device
            )
            question_input = text_input

        targets = text_input.input_ids.masked_fill(
            text_input.input_ids == tokenizer.pad_token_id, -100
        )

        answer_output = model(
            image=image,
            text=text_input,
            labels=targets,
            return_dict=True,
            mode="train",
            question=question_input,
        )

        loss = answer_output.loss
        loss = loss.sum() / image.size(0)
        loss = loss * lm_loss_weight

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if prompt_lr is not None:
            metric_logger.update(prompt_lr=optimizer.param_groups[1]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    accelerator.print("Averaged stats:", metric_logger.global_avg())
    return {
        k: "{:.3f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
    }


def generate_to_inference_save_mode(out):

    # (number of batches, number of inference/generated tokens, number of layers, bsxbeams, l (promptlen for idx0 and 1 for others), dim)
    # (nbs, l, 33, bs, 1 or 11, dim) -> (nbs*bs, 33, 1 + l or 11 + l, dim)

    hidden_states = [
        [o.cpu() for o in O] for O in out["hidden_states"]
    ]  # out['hidden_states'].cpu() #[out['hidden_states'][j][i].cpu() for j in range(len(out['hidden_states']))]

    data = [{"hidden_states": hidden_states}]
    # data = test['all_hidden_states']
    data1_ = []

    bs = hidden_states[0][0].shape[0]
    print(bs)
    for i in range(len(data)):
        for j in range(bs):
            item = [
                [D[j] for D in L] for L in data[i]["hidden_states"]
            ]  # l, 33, 1 or 11, dim
            data1_.append(item)  # (nbs*bs, l, 33, 1  or 11 , dim)
    print(len(data1_), len(data1_[0]), len(data1_[0][0]))
    data2_ = []
    for d in data1_:
        item = [[l[i] for l in d] for i in range(len(d[0]))]  # 33, l, 1 or 11, dim
        item_ = [torch.cat(l) for l in item]  # 33, l+1 or l+11, dim
        data2_.append(item_)  # (nbs*bs, 33, 1 + l or 11 + l, dim)
    print(len(data2_), len(data2_[0]))
    return data2_


@torch.no_grad()
def evaluation(
    model,
    data_loader,
    tokenizer,
    device,
    config,
    accelerator=None,
    max_length=30,
    only_main=False,
    noise_modality=False,
):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Generate Caption test result:"
    print_freq = 50

    predictions = []
    targets = []

    pad_token = tokenizer.pad_token
    eos_token = tokenizer.eos_token
    bos_token = tokenizer.bos_token

    num_beams = config.get("num_beams", 1)
    do_sample = config.get("do_sample", True)
    accelerator.print(
        "num_beams",
        num_beams,
        "do_sample",
        do_sample,
        "max_length",
        max_length,
        "eos_token",
        eos_token,
    )

    inference_mode = config.get("inference_mode", "generate")
    save_hidden_states = config.get("save_hidden_states", False)
    output_intermediate_hidden_states = config.get(
        "output_intermediate_hidden_states", False
    )
    keys_to_save = config.get("keys_to_save", None)

    if save_hidden_states:
        save_kwargs = {
            "output_hidden_states": True,
            "output_attentions": True,
            "return_dict_in_generate": True,
        }
        all_hidden_states = []
    else:
        save_kwargs = {
            "output_hidden_states": False,
            "output_attentions": False,
            "return_dict_in_generate": False,
        }

    instruction = config.get("instruction", "")

    task = data_loader.task
    if task == "vqa":
        quesid2ans = {}

    save_only_image = config.get("save_only_image", False)

    image_feats = []
    attention_masks = []

    accelerator.print(f"Evaluating with noise_modality: {noise_modality}")

    for n, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        image = batch["images"].to(device, non_blocking=True)

        if task == "vqa":
            text = [f"Question:{q} Answer:" for q in batch["sent"]]
        else:
            if save_only_image and inference_mode != "generate":
                if isinstance(batch["targets"][0], list):
                    text = [f"{q[0]}" for q in batch["targets"]]
                else:
                    text = [f"{q}" for q in batch["targets"]]
            else:
                if isinstance(instruction, list):
                    inst = random.choice(instruction)
                else:
                    inst = instruction
                text = [inst for t in image]
        text_input = tokenizer(text, padding="longest", return_tensors="pt").to(device)
        question_input = text_input

        out = model(
            image=image,
            text=text_input,
            mode=inference_mode,
            return_dict=True,
            max_length=max_length,
            do_sample=do_sample,
            num_beams=num_beams,
            question=question_input,
            only_image=save_only_image,
            noise_modality=noise_modality,
            **save_kwargs,
        )

        if save_only_image:
            for o, t, a in zip(out[0][0], out[1], out[2]):  # over batch
                image_feats.append(
                    {
                        "image_feat": o.half().cpu(),
                        "text_embed": t.half().cpu(),
                        "attention_mask": a.cpu(),
                    }
                )  # (10, dim)
        elif save_hidden_states:
            bs = len(text)  # out['vision_states_after'][0].shape[0]
            # out['hidden_states'] num_layer, bs, l, dim
            attention_masks = text_input.attention_mask
            if inference_mode == "evaluate":
                for i in range(bs):

                    if output_intermediate_hidden_states:
                        hidden_states = [
                            out["hidden_states"][j][0][i].cpu()
                            for j in range(len(out["hidden_states"]))
                        ]
                        intermediate_hidden_atts = [
                            out["hidden_states"][j][1][1][i].cpu()
                            for j in range(len(out["hidden_states"]))
                        ]
                        intermediate_hidden_atts_res = [
                            out["hidden_states"][j][1][2][i].cpu()
                            for j in range(len(out["hidden_states"]))
                        ]
                        intermediate_hidden_act = [
                            out["hidden_states"][j][1][4][i].cpu()
                            for j in range(len(out["hidden_states"]))
                        ]
                        intermediate_hidden_fc2 = [
                            out["hidden_states"][j][1][5][i].cpu()
                            for j in range(len(out["hidden_states"]))
                        ]
                        intermediate_hidden_fc2_res = [
                            out["hidden_states"][j][1][6][i].cpu()
                            for j in range(len(out["hidden_states"]))
                        ]

                        intermediate_hidden_preatts = [
                            out["hidden_states"][j][1][0][i].cpu()
                            for j in range(len(out["hidden_states"]))
                        ]
                        intermediate_hidden_prefc = [
                            out["hidden_states"][j][1][3][i].cpu()
                            for j in range(len(out["hidden_states"]))
                        ]
                    else:
                        hidden_states = [
                            out["hidden_states"][j][i].cpu()
                            for j in range(len(out["hidden_states"]))
                        ]
                        intermediate_hidden_atts = []
                        intermediate_hidden_act = []
                        intermediate_hidden_fc2 = []
                        intermediate_hidden_fc2_res = []
                        intermediate_hidden_atts_res = []

                        intermediate_hidden_preatts = []
                        intermediate_hidden_prefc = []

                    hidden_states = {
                        "hidden_states": hidden_states,
                        "intermediate_hidden_atts": intermediate_hidden_atts,
                        "intermediate_hidden_act": intermediate_hidden_act,
                        "intermediate_hidden_fc2": intermediate_hidden_fc2,
                        "intermediate_hidden_atts_res": intermediate_hidden_atts_res,
                        "intermediate_hidden_fc2_res": intermediate_hidden_fc2_res,
                        "intermediate_hidden_preatts": intermediate_hidden_preatts,
                        "intermediate_hidden_prefc": intermediate_hidden_prefc,
                        "attention_masks": attention_masks[i].cpu(),
                    }

                    if keys_to_save is not None:
                        for k in hidden_states.keys():
                            if k not in keys_to_save:
                                hidden_states[k] = []

                    all_hidden_states.append(hidden_states)

            else:
                os = generate_to_inference_save_mode(out)
                for i in range(bs):

                    # hidden_states: (number of inference/generated tokens, number of layers, bsxbeams, l (promptlen for idx0 and 1 for others), dim)
                    hidden_states = os[
                        i
                    ]  # [[o.cpu() for o in O] for O in out['hidden_states']] # out['hidden_states'].cpu() #[out['hidden_states'][j][i].cpu() for j in range(len(out['hidden_states']))]

                    intermediate_hidden_atts = []
                    intermediate_hidden_act = []
                    intermediate_hidden_fc2 = []
                    intermediate_hidden_fc2_res = []
                    intermediate_hidden_atts_res = []

                    intermediate_hidden_preatts = []
                    intermediate_hidden_prefc = []

                    hidden_states = {
                        "hidden_states": hidden_states,
                        "intermediate_hidden_atts": intermediate_hidden_atts,
                        "intermediate_hidden_act": intermediate_hidden_act,
                        "intermediate_hidden_fc2": intermediate_hidden_fc2,
                        "intermediate_hidden_atts_res": intermediate_hidden_atts_res,
                        "intermediate_hidden_fc2_res": intermediate_hidden_fc2_res,
                        "intermediate_hidden_preatts": intermediate_hidden_preatts,
                        "intermediate_hidden_prefc": intermediate_hidden_prefc,
                        "attention_masks": attention_masks[i].cpu(),
                    }

                    if keys_to_save is not None:
                        for k in hidden_states.keys():
                            if k not in keys_to_save:
                                hidden_states[k] = []

                    all_hidden_states.append(hidden_states)
        else:
            if save_hidden_states:
                hidden_states = {
                    "hidden_states": [
                        [o.half().cpu() for o in O] for O in out["hidden_states"]
                    ],
                    "vision_states_before": [
                        o.half().cpu() for o in out["vision_states_before"]
                    ],
                    "vision_states_after": [
                        o.half().cpu() for o in out["vision_states_after"]
                    ],
                    "vision_states_connector": [
                        out["vision_states_connector"][-1][j][i].half().cpu()
                        for j in range(len(out["vision_states_connector"][-1]))
                    ],
                }
                all_hidden_states.append(hidden_states)

                out = out["sequences"]  # outputs[:, len(input_ids[0]) :]

            out_decode = []

            if task == "vqa":
                question_id = batch["question_ids"]
            for i, o in enumerate(out):
                try:
                    res = tokenizer.decode(o)
                    response = res.split(bos_token)[1]  # skip_special_tokens=True
                    response = (
                        response.replace(pad_token, "").replace(eos_token, "").strip()
                    )

                except (TypeError, IndexError):
                    accelerator.print(o)
                    response = " "

                out_decode.append(response)

                if task == "vqa":
                    try:
                        ques_id = int(question_id[i])
                    except ValueError:
                        ques_id = question_id[i]

                    pattern = "|".join(map(re.escape, ["Answer", "Assistant"]))
                    try:
                        response = re.split(pattern, response, flags=re.IGNORECASE)[1]
                    except:
                        pass
                    for word in ["Answer", "Question", "Assistant", "Human", ":"]:
                        response = response.replace(word, "")

                    try:
                        max_ans_len = max([len(a) for a in batch["all_answers"][i]])
                        response = response[:max_ans_len]
                    except:
                        max_ans_len = -1
                        response = response.split(" ")[0]

                    quesid2ans[ques_id] = response

        if not save_hidden_states:  # (save_only_image or inference_mode != 'generate'):

            if save_hidden_states and n % 10 == 0:
                print(n, response)

            predictions.extend(out_decode)

            if "targets" in batch:
                targets.extend(batch["targets"])

    if save_only_image:
        return image_feats
    if save_hidden_states:
        return all_hidden_states

    if (
        accelerator.state.num_processes > 1
        and dist.get_world_size() > 1
        and not only_main
    ):
        print("gather from different gpus")
        gather_predictions = [None for _ in range(dist.get_world_size())]
        gather_targets = [None for _ in range(dist.get_world_size())]

        dist.all_gather_object(gather_predictions, predictions)
        dist.all_gather_object(gather_targets, targets)

        predictions = list(chain(*gather_predictions))
        targets = list(chain(*gather_targets))

    evaluator = data_loader.evaluator

    if data_loader.task == "vqa":
        try:
            eval_results = evaluator.evaluate_raw(quesid2ans)

        except AttributeError:
            eval_results = {}

        topk_score = evaluator.evaluate(quesid2ans)
        eval_results["topk_score"] = topk_score
        if "overall" not in eval_results:
            eval_results["overall"] = topk_score
    else:
        eval_results = evaluator.evaluate(predictions, targets)

    wandb_log_dict = {}

    for score_name, score in eval_results.items():
        wandb_log_dict[f"Valid/{score_name}"] = score

    accelerator.print(wandb_log_dict)

    if save_hidden_states:
        return wandb_log_dict, all_hidden_states

    return wandb_log_dict


@torch.no_grad()
def evaluation_stats(
    model,
    data_loader,
    tokenizer,
    device,
    config,
    accelerator=None,
    max_length=30,
    only_main=False,
    verbose=False,
    mode="generate",
):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Generate Caption test result:"
    print_freq = 50

    pad_token = tokenizer.pad_token
    eos_token = tokenizer.eos_token

    num_beams = config.get("num_beams", 1)
    do_sample = config.get("do_sample", True)

    if mode == "evaluate":
        num_beams = 1
        do_sample = False
    accelerator.print(
        "num_beams", num_beams, "do_sample", do_sample, "max_length", max_length
    )

    save_kwargs = {
        "output_hidden_states": False,
        "output_attentions": False,
        "return_dict_in_generate": False,
    }
    instruction = config.get("instruction", "")

    task = data_loader.task

    calculate_flops_pipline = CalFlopsPipline(
        model=model.module.model_text,
        include_backPropagation=False,
        compute_bp_factor=2.0,
    )

    calculate_flops_pipline.start_flops_calculate(
        ignore_list=None
    )  # to add vision encoder and connector ?

    all_times = []
    all_flops = []
    all_macs = []
    all_lens = []
    for n, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        image = batch["images"].to(device, non_blocking=True)
        bs = image.shape[0]

        if mode == "evaluate":
            if "sent" in batch:
                sent = batch["sent"]
            else:
                sent = batch["targets"]
                if isinstance(sent[0], list):
                    sent = [s[0] for s in sent]

            if task == "vqa":
                text = [
                    f"Question:{q} Answer:{a.replace('[SEP]','')}"
                    for q, a in zip(sent, batch["answers"])
                ]
            else:
                if isinstance(instruction, list):
                    inst = random.choice(instruction)
                else:
                    inst = instruction
                text = [inst for t in range(bs)]
        else:
            if task == "vqa":
                text = [f"Question:{q} Answer:" for q in batch["sent"]]
            else:
                if isinstance(instruction, list):
                    inst = random.choice(instruction)
                else:
                    inst = instruction
                text = [inst for t in range(bs)]

        text_input = tokenizer(text, padding="longest", return_tensors="pt").to(device)
        question_input = text_input

        out, t = model(
            image=image,
            text=text_input,
            mode=mode,
            return_dict=True,
            max_length=max_length,
            do_sample=do_sample,
            num_beams=num_beams,
            question=question_input,
            verbose=verbose,
            return_time=True,
            **save_kwargs,
        )

        all_times.append(t)

        flops = calculate_flops_pipline.get_total_flops() / bs
        macs = calculate_flops_pipline.get_total_macs() / bs

        all_flops.append(flops)
        all_macs.append(macs)

        lens = []
        if mode != "evaluate":
            for i, o in enumerate(out):
                res = tokenizer.decode(o)
                response = (
                    res.replace(pad_token, "")
                    .replace("</s>", "")
                    .replace(eos_token, "")
                    .strip()
                )  # skip_special_tokens=True
                lens.append(len(response))
            all_lens.append(sum(lens) / len(lens))
        else:
            all_lens = [0]

        calculate_flops_pipline.reset_flops_calculate()
        if n > 50:
            break

    torch.cuda.synchronize()

    if (
        accelerator.state.num_processes > 1
        and dist.get_world_size() > 1
        and not only_main
    ):
        gather_flops = [None for _ in range(dist.get_world_size())]
        gather_macs = [None for _ in range(dist.get_world_size())]
        gather_times = [None for _ in range(dist.get_world_size())]
        gather_lens = [None for _ in range(dist.get_world_size())]

        dist.all_gather_object(gather_flops, all_flops)
        dist.all_gather_object(gather_macs, all_macs)
        dist.all_gather_object(gather_times, all_times)
        dist.all_gather_object(gather_lens, all_lens)

        all_flops = list(chain(*gather_flops))
        all_macs = list(chain(*gather_macs))
        all_times = list(chain(*gather_times))
        all_lens = list(chain(*gather_lens))

    calculate_flops_pipline.end_flops_calculate()

    l = len(all_flops)
    all_flops, all_macs, all_times = (
        sum(all_flops) / l,
        sum(all_macs) / l,
        sum(all_times) / l,
    )
    all_lens = sum(all_lens) / l

    all_flops = float(remove_letters(number_to_string(all_flops, precision=4)))
    all_macs = float(remove_letters(number_to_string(all_macs, precision=4)))

    print(
        f"Length: {all_lens}, Latency/Time: {all_times:.4f} (ms), FLOPS: {all_flops}, MACS: {all_macs}"
    )

    return all_flops, all_macs, round(all_times, 4), all_lens


def main(args, config):
    if "XDG_CACHE_HOME" in os.environ:
        os.environ["TORCH_HOME"] = os.environ["XDG_CACHE_HOME"] + "/torch"
    else:
        os.environ["TORCH_HOME"] = "~/.cache/torch"

    args.distributed = False

    accelerator = Accelerator()

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config["schedular"]["epochs"]
    warmup_steps = config["schedular"]["warmup_epochs"]

    accelerator.print(args, config)
    #### Dataset ####

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
    else:
        num_tasks = None
        global_rank = None

    train_split = config.get("train_split", "train")
    val_split = config.get("val_split", "val")
    test_split = config.get("test_split", "test")

    #########
    num_workers = config.get("num_workers", 4)
    train_topk = config.get("train_topk", -1)
    valid_topk = config.get("valid_topk", -1)
    test_topk = config.get("test_topk", -1)
    if args.test_topk > 0:
        test_topk = args.test_topk

    if args.debug:
        valid_topk = 0.1

    data_dir = args.data_dir

    config["image_res"] = args.image_size

    vision_model_name = config.get("vision_model_name", args.vision_model)

    dataset_name = args.dataset_name
    if dataset_name == "vqav2":
        train_split = config.get("train_split", "karpathy_train")
        val_split = config.get("val_split", "karpathy_val")
        test_split = config.get("test_split", "karpathy_test")

    test_dataset_name = (
        args.test_dataset_name if args.test_dataset_name is not None else dataset_name
    )

    accelerator.print(
        f"dataset_name: {dataset_name}, test_dataset_name: {test_dataset_name}"
    )

    if dataset_name == "vqav2":
        get_loader_ = get_loader_vqa
    elif dataset_name == "gqa":
        get_loader_ = get_loader_gqa
    elif dataset_name == "okvqa":
        get_loader_ = get_loader_okvqa
    elif dataset_name == "textvqa":
        get_loader_ = get_loader_textvqa
    elif dataset_name == "aokvqa":
        get_loader_ = get_loader_aokvqa
    elif dataset_name == "msrvtt":
        get_loader_ = get_loader_msrvtt
    elif dataset_name == "msrvtqa" or dataset_name == "msvd":
        get_loader_ = get_loader_msrvttqa
    elif dataset_name == "audiocaps" or dataset_name == "clotho":
        get_loader_ = get_loader_audiocaps
    elif dataset_name == "clotho_aqa":
        get_loader_ = get_loader_clothoaqa
    else:
        get_loader_ = get_loader

    if test_dataset_name == "vqav2":
        get_loader_eval = get_loader_vqa
    elif test_dataset_name == "gqa":
        get_loader_eval = get_loader_gqa
    elif test_dataset_name == "okvqa":
        get_loader_eval = get_loader_okvqa
    elif test_dataset_name == "textvqa":
        get_loader_eval = get_loader_textvqa
    elif test_dataset_name == "aokvqa":
        get_loader_eval = get_loader_aokvqa
    elif test_dataset_name == "msrvtt":
        get_loader_eval = get_loader_msrvtt
    elif test_dataset_name == "msrvtqa" or test_dataset_name == "msvd":
        get_loader_eval = get_loader_msrvttqa
    elif test_dataset_name == "audiocaps" or test_dataset_name == "clotho":
        get_loader_eval = get_loader_audiocaps
    elif test_dataset_name == "clotho_aqa":
        get_loader_eval = get_loader_clothoaqa
    else:
        get_loader_eval = get_loader

    image_size = args.image_size
    use_data_augmentation = True
    data_json_dir = args.data_json_dir

    if args.test_split is not None:
        test_split = args.test_split
    if args.train_split is not None:
        train_split = args.train_split
    if args.val_split is not None:
        val_split = args.val_split

    test_batch_size = config["batch_size_test"]
    if args.test_batch_size:
        test_batch_size = args.test_batch_size
    test_data_dir = args.test_data_dir if args.test_data_dir is not None else data_dir

    accelerator.print(
        f"test_dataset_name:{test_dataset_name} test_data_dir:{test_data_dir}, test_split:{test_split}, dataset_name:{dataset_name}, data_dir:{data_dir}, train_split:{train_split}"
    )

    if any([k in dataset_name for k in ["vqav2", "gqa", "okvqa", "textvqa", "aokvqa"]]):

        train_loader, train_dataset = get_loader_(
            split=train_split,
            mode="train",
            batch_size=config["batch_size_train"],
            distributed=args.distributed,
            workers=num_workers,
            topk=train_topk,
            data_dir=data_dir,
            local_rank=global_rank,
            world_size=num_tasks,
            verbose=True,
            image_size=image_size,
            use_data_augmentation=use_data_augmentation,
        )

    elif any([k in dataset_name for k in ["msrvtt", "msrvtqa", "msvd"]]):

        num_frames = config.get("num_frames", 8)

        num_tries = config.get("num_tries", 2)
        sample_type = config.get(
            "sample_type", "rand"
        )  # fps1 (1 frame per second with cropping to the max number of frames )
        image_size = config.get("image_size", 224)
        use_data_augmentation = config.get("use_data_augmentation", True)

        as_images = False if "timesformer" in vision_model_name else True

        train_loader, train_dataset = get_loader_(
            split=train_split,
            mode="train",
            batch_size=config["batch_size_train"],
            distributed=args.distributed,
            workers=num_workers,
            topk=train_topk,
            data_dir=data_dir,
            local_rank=global_rank,
            world_size=num_tasks,
            verbose=True,
            num_frames=num_frames,
            as_images=as_images,
            num_tries=num_tries,
            sample_type=sample_type,
            image_size=image_size,
            use_data_augmentation=use_data_augmentation,
        )

    elif any([k in dataset_name for k in ["audiocaps", "clotho", "clotho_qa"]]):

        melbins = config.get("melbins", 64)
        target_length = config.get("target_length", 1024)
        num_tries = config.get("num_tries", 2)
        freqm_p = config.get("freqm_p", 24)
        timem_p = config.get("timem_p", 96)
        skip_norm = config.get("skip_norm", False)
        norm_mean = config.get("norm_mean", -4.2677393)
        norm_std = config.get("norm_std", 4.5689974)
        noise = config.get("noise", False)

        processor_audio = None

        train_loader, train_dataset = get_loader_(
            split=train_split,
            mode="train",
            batch_size=config["batch_size_train"],
            distributed=args.distributed,
            workers=num_workers,
            topk=train_topk,
            data_dir=data_dir,
            local_rank=global_rank,
            world_size=num_tasks,
            verbose=True,
            melbins=melbins,
            target_length=target_length,
            num_tries=num_tries,
            freqm_p=freqm_p,
            timem_p=timem_p,
            skip_norm=skip_norm,
            norm_mean=norm_mean,
            norm_std=norm_std,
            noise=noise,
            processor=processor_audio,
        )

    else:

        train_loader, train_dataset = get_loader_(
            split="train",
            mode="train",
            batch_size=config["batch_size_train"],
            distributed=args.distributed,
            workers=num_workers,
            topk=train_topk,
            data_dir=data_dir,
            local_rank=global_rank,
            world_size=num_tasks,
            verbose=True,
            image_size=image_size,
            use_data_augmentation=use_data_augmentation,
        )

    if any(
        [k in test_dataset_name for k in ["vqav2", "gqa", "okvqa", "textvqa", "aokvqa"]]
    ):

        accelerator.print(f"Building val loader")
        val_loader, val_dataset = get_loader_eval(
            split=val_split,
            mode="val",
            batch_size=test_batch_size,
            distributed=False,
            workers=4,
            topk=valid_topk,
            data_dir=test_data_dir,
            local_rank=global_rank,
            world_size=num_tasks,
            verbose=True,
            image_size=image_size,
            use_data_augmentation=use_data_augmentation,
        )
        accelerator.print("# len val loader:", len(val_loader))

        accelerator.print(f"Building test loader")
        test_loader, test_dataset = get_loader_eval(
            split=test_split,
            mode="val",
            batch_size=test_batch_size,
            distributed=False,
            workers=4,
            topk=test_topk,
            data_dir=test_data_dir,
            local_rank=global_rank,
            world_size=num_tasks,
            verbose=True,
            image_size=image_size,
            use_data_augmentation=use_data_augmentation,
        )

    elif any([k in test_dataset_name for k in ["msrvtt", "msrvtqa", "msvd"]]):
        metrics = ["CIDEr", "BLEU"]

        num_frames = config.get("num_frames", 8)

        num_tries = config.get("num_tries", 2)
        sample_type = config.get(
            "sample_type", "rand"
        )  # fps1 (1 frame per second with cropping to the max number of frames )
        image_size = config.get("image_size", 224)
        use_data_augmentation = config.get("use_data_augmentation", True)

        as_images = False if "timesformer" in vision_model_name else True

        accelerator.print(f"Building val loader")
        val_loader, val_dataset = get_loader_eval(
            split=val_split,
            mode="val",
            batch_size=test_batch_size,
            distributed=False,
            workers=4,
            topk=valid_topk,
            data_dir=test_data_dir,
            local_rank=global_rank,
            world_size=num_tasks,
            verbose=True,
            num_frames=num_frames,
            as_images=as_images,
            num_tries=num_tries,
            sample_type=sample_type,
            image_size=image_size,
            use_data_augmentation=use_data_augmentation,
            metrics=metrics,
        )
        accelerator.print("# len val loader:", len(val_loader))

        accelerator.print(f"Building test loader")
        test_loader, test_dataset = get_loader_eval(
            split=test_split,
            mode="val",
            batch_size=test_batch_size,
            distributed=False,
            workers=4,
            topk=test_topk,
            data_dir=test_data_dir,
            local_rank=global_rank,
            world_size=num_tasks,
            verbose=True,
            num_frames=num_frames,
            as_images=as_images,
            num_tries=num_tries,
            sample_type=sample_type,
            image_size=image_size,
            use_data_augmentation=use_data_augmentation,
            metrics=metrics,
        )

    elif any([k in dataset_name for k in ["audiocaps", "clotho", "clotho_qa"]]):

        metrics = ["CIDEr", "BLEU"]
        melbins = config.get("melbins", 64)
        target_length = config.get("target_length", 1024)
        num_tries = config.get("num_tries", 2)
        freqm_p = config.get("freqm_p", 24)
        timem_p = config.get("timem_p", 96)
        skip_norm = config.get("skip_norm", False)
        norm_mean = config.get("norm_mean", -4.2677393)
        norm_std = config.get("norm_std", 4.5689974)
        noise = config.get("noise", False)

        processor_audio = None

        accelerator.print(f"Building val loader")
        val_loader, val_dataset = get_loader_eval(
            split=val_split,
            mode="val",
            batch_size=test_batch_size,
            distributed=False,
            workers=4,
            topk=valid_topk,
            data_dir=test_data_dir,
            local_rank=global_rank,
            world_size=num_tasks,
            verbose=True,
            melbins=melbins,
            target_length=target_length,
            num_tries=num_tries,
            freqm_p=freqm_p,
            timem_p=timem_p,
            skip_norm=skip_norm,
            norm_mean=norm_mean,
            norm_std=norm_std,
            noise=noise,
            processor=processor_audio,
            metrics=metrics,
        )
        accelerator.print("# len val loader:", len(val_loader))

        accelerator.print(f"Building test loader")
        test_loader, test_dataset = get_loader_eval(
            split=test_split,
            mode="val",
            batch_size=test_batch_size,
            distributed=False,
            workers=4,
            topk=test_topk,
            data_dir=test_data_dir,
            local_rank=global_rank,
            world_size=num_tasks,
            verbose=True,
            melbins=melbins,
            target_length=target_length,
            num_tries=num_tries,
            freqm_p=freqm_p,
            timem_p=timem_p,
            skip_norm=skip_norm,
            norm_mean=norm_mean,
            norm_std=norm_std,
            noise=noise,
            processor=processor_audio,
            metrics=metrics,
        )

    else:
        metrics = ["CIDEr", "BLEU"]
        accelerator.print(f"Building val loader")
        val_loader, val_dataset = get_loader_eval(
            split=val_split,
            mode="val",
            batch_size=test_batch_size,
            distributed=False,
            workers=4,
            topk=valid_topk,
            data_dir=test_data_dir,
            local_rank=global_rank,
            world_size=num_tasks,
            verbose=True,
            image_size=image_size,
            use_data_augmentation=use_data_augmentation,
            metrics=metrics,
        )
        accelerator.print("# len val loader:", len(val_loader))

        accelerator.print(f"Building test loader")
        test_loader, test_dataset = get_loader_eval(
            split=test_split,
            mode="val",
            batch_size=test_batch_size,
            distributed=False,
            workers=4,
            topk=test_topk,
            data_dir=test_data_dir,
            local_rank=global_rank,
            world_size=num_tasks,
            verbose=True,
            image_size=image_size,
            use_data_augmentation=use_data_augmentation,
            metrics=metrics,
        )

    # if gpu == 0:
    accelerator.print("# len train loader:", len(train_loader))
    accelerator.print("# len test loader:", len(test_loader))

    #### Model ####
    accelerator.print("Creating model")

    start_layer_idx = config.get("start_layer_idx", 0)
    end_layer_idx = config.get("end_layer_idx", 0)

    config["output_hidden_states"] = args.save_hidden_states
    config["noise_modality"] = args.noise_modality

    config["output_intermediate_hidden_states"] = args.output_intermediate_hidden_states
    config["keys_to_save"] = (
        list(args.keys_to_save.split(",")) if args.keys_to_save is not None else None
    )
    accelerator.print(f"Only save the following keys: {config['keys_to_save']}")

    model = ePALMv2(
        opt_model_name=args.text_model,
        vision_model_name=vision_model_name,
        start_layer_idx=start_layer_idx,
        end_layer_idx=end_layer_idx,
        return_hidden_state_vision=True,
        config=config,
    )

    model = model.to(device)

    tokenizer_name = config.get("tokenizer_name", args.text_model)

    if "opt" in tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.text_model, use_fast=False, local_files_only=True, padding_side="left"
        )
    elif any([k in tokenizer_name for k in ["llama", "vicuna"]]):
        tokenizer = LlamaTokenizer.from_pretrained(
            args.text_model, local_files_only=True, padding_side="left"
        )

        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model.model_text,
            )

            model.model_text.config.pad_token_id = tokenizer.pad_token_id

            accelerator.print(
                f"resize embeding to add pad token {tokenizer.pad_token_id}"
            )

    else:
        raise NotImplemented

    arg_opt = utils.AttrDict(config["optimizer"])
    optimizer = create_optimizer(arg_opt, model, config=config)

    if hasattr(arg_opt, "prompt_lr") and arg_opt.prompt_lr is not None:
        accelerator.print(
            "\tInitial other params params lr: %f" % optimizer.param_groups[0]["lr"]
        )
        accelerator.print(
            "\tInitial prompt params lr: %f" % optimizer.param_groups[1]["lr"]
        )

    arg_sche = utils.AttrDict(config["schedular"])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    best_epoch = 0
    best_valid = 0

    if "rescaleconv" in vision_model_name:
        exclude_list.remove("model_vision")

    if args.checkpoint:

        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint["model"]
        msg = model.load_state_dict(state_dict, strict=False)
        msg = filter_msg(msg, exclude_list)

        accelerator.print("load checkpoint from %s" % args.checkpoint)
        accelerator.print(msg)

        if args.resume:
            model = model.to(device)
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            accelerator.print(checkpoint.keys())
            for p in optimizer.param_groups:  # not necessay after torch 1.12.1
                p["capturable"] = True
        if "best_valid" in checkpoint:
            best_valid = checkpoint["best_valid"]
            best_epoch = checkpoint["best_epoch"]
            accelerator.print(
                "load best valid {} at epoch {}".format(best_valid, best_epoch)
            )

    freeze_whole_model(model)
    unfreeze_parameters(model, config, accelerator=accelerator)

    if "only_patch_proj" in vision_model_name and "unfreeze" in vision_model_name:
        accelerator.print("unfreeze only_patch_proj ...")
        model.model_vision.float()
        model.model_vision.conv1.requires_grad_(True)
        model.model_vision.positional_embedding.requires_grad_(True)
        model.model_vision.ln_pre.requires_grad_(True)
        model.model_vision.class_embedding.requires_grad_(True)

    print_trainable_params_percentage(model, accelerator=accelerator)

    # pruning
    if args.sparsity_ratio != 0:
        if "block" in args.prune_method and not accelerator.is_main_process:
            pass
        else:

            # Handling n:m sparsity
            prune_n, prune_m = 0, 0
            if args.sparsity_type != "unstructured":
                assert (
                    args.sparsity_ratio == 0.5
                ), "sparsity ratio must be 0.5 for structured N:M sparsity"
                prune_n, prune_m = map(int, args.sparsity_type.split(":"))

            hidden_size = model.model_text.config.hidden_size
            task = train_loader.task
            prune_loader = train_loader
            print(f"Pruninng data tas: {prune_loader.task}")

            seqlen = 128
            layers = model.get_llm_layers()

            use_cache = model.model_text.config.use_cache
            model.model_text.config.use_cache = False

            print(
                f"Pruning mode: {args.prune_method}, sparsity_ratio: {args.sparsity_ratio}"
            )
            print("pruning starts")

            # for opt
            if args.prune_layer_name == "ffn":
                layer_name = (
                    ["up_proj", "down_proj", "gate_proj"]
                    if any([k in args.text_model for k in ["llama", "vicuna"]])
                    else ["fc1", "fc2"]
                )
            elif args.prune_layer_name == "sa":
                layer_name = ["self_attn"]
            else:
                layer_name = None

            is_llama = any([k in args.text_model for k in ["llama", "vicuna"]])

            pruning_verbose = not args.only_prune

            per_layer_type = args.per_layer_type

            sparsity_list = None
            ffn_to_sa_sparsity = 1
            sa_sparsity_list, ffn_sparsity_list = None, None

            sparsity_list = None
            ffn_to_sa_sparsity = 1

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                if args.prune_method == "wanda":
                    prune_wanda(
                        args,
                        model,
                        tokenizer,
                        layers,
                        device=device,
                        prune_n=prune_n,
                        prune_m=prune_m,
                        global_batch_size=args.num_calibration_data,
                        task=task,
                        dataloader=prune_loader,
                        seqlen=seqlen,
                        hidden_size=hidden_size,
                        nb_steps=args.prune_steps,
                        start_layer=args.start_layer,
                        end_layer=args.end_layer,
                        layer_name=layer_name,
                        layer_interval=args.layer_interval,
                        save_path=args.prune_save_path,
                        only_prompt=args.only_prompt,
                        only_text=args.only_text,
                        with_answers=args.with_answers,
                        is_llama=is_llama,
                        only_prune=args.only_prune,
                        verbose=pruning_verbose,
                        sparsity_list=sparsity_list,
                        ffn_to_sa_sparsity=ffn_to_sa_sparsity,
                        ffn_sparsity_list=ffn_sparsity_list,
                        sa_sparsity_list=sa_sparsity_list,
                        accelerator=accelerator,
                        noise_modality=args.noise_modality,
                    )
                elif args.prune_method == "given_mask":
                    accelerator.print(f"load mask from: {args.mask_path}")
                    prune_given_mask(
                        layers,
                        start_layer=args.start_layer,
                        end_layer=args.end_layer,
                        layer_name=layer_name,
                        layer_interval=args.layer_interval,
                        mask_path=args.mask_path,
                        ex_sparsity=args.ex_sparsity,
                    )

                elif args.prune_method == "magnitude":
                    prune_magnitude(
                        args,
                        model,
                        tokenizer,
                        layers,
                        device,
                        prune_n=prune_n,
                        prune_m=prune_m,
                        save_path=args.prune_save_path,
                        only_prune=args.only_prune,
                        per_output=args.per_output,
                    )
                model.model_text.config.use_cache = use_cache

                accelerator.print("*" * 30)
                sparsity_ratio = check_sparsity(
                    layers,
                    accelerator=accelerator,
                    show_layers="block" in args.prune_method,
                )
                accelerator.print(f"sparsity sanity check {sparsity_ratio:.4f}")
                accelerator.print("*" * 30)

    if args.only_prune:
        return

    unimodal_attention_mask = args.unimodal_attention_mask
    stop_unimodal_attention_mask = args.stop_unimodal_attention_mask
    if unimodal_attention_mask:
        if any([k in args.text_model for k in ["llama", "vicuna"]]):
            print("unimodal_attention_mask llama utils ...")
            from pruning.skipping_llama import (
                forward_llama, forward_llama_decoder, forward_llama_layer, forward_llama_mlp)

            model.model_text.model.forward = partial(
                forward_llama_decoder, self=model.model_text.model
            )
            model.model_text.forward = partial(
                forward_llama,
                self=model.model_text,
                unimodal_attention_mask=unimodal_attention_mask,
                stop_unimodal_attention_mask=stop_unimodal_attention_mask,
            )
        else:
            raise NotImplemented

    ## skipping
    if args.skipping_mode or args.output_intermediate_hidden_states:

        skip_interval = args.skip_interval

        len_layers = len(model.get_llm_layers())
        if isinstance(skip_interval, str) and "," in skip_interval:
            skip_interval = list(map(int, skip_interval.split(",")))
            len_skip_int = len(skip_interval)

            step = len_layers // len_skip_int

            tmp = [
                skip_interval[min(idx // step, len_skip_int - 1)]
                for idx in range(len_layers)
            ]

            skip_intervals = []
            for idx in range(len_layers):
                if idx % tmp[idx] == 0:
                    skip_intervals.append(1)  # skip
                else:
                    skip_intervals.append(32)
            skip_interval = skip_intervals
        else:
            skip_interval = int(skip_interval)

        layers_interval = (
            list(range(0, len_layers, args.layers_interval))
            if args.layers_interval > 0
            else []
        )

        accelerator.print(
            f"Skipping mode: {args.skip_mode}, skip_interval: {skip_interval}, layers_interval: {layers_interval}, exit layer: {args.exit_layer}, start_drop_layer: {args.start_drop_layer}"
        )

        from pruning.skipping_opt import compute_skipped_blocks

        num_skips = compute_skipped_blocks(
            skip_mode=args.skip_mode,
            skip_interval=skip_interval,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            layers_interval=layers_interval,
            exit_layer=args.exit_layer,
            start_drop_layer=args.start_drop_layer,
            num_layers=32,
        )

        accelerator.print(f"Number of skips: {num_skips}")

        prompt_len = config.get("prompt_len", 10)
        accelerator.print(f"prompt_len: {prompt_len}")

        if any([k in args.text_model for k in ["llama", "vicuna"]]):
            print("skipping llama utils ...")
            from pruning.skipping_llama import (
                forward_llama, forward_llama_decoder, forward_llama_layer, forward_llama_mlp)

            for layer in model.get_llm_layers():
                layer.forward = partial(
                    forward_llama_layer, self=layer
                )  # partial(forward_opt_layer, self=layer)
                layer.mlp.forward = partial(forward_llama_mlp, self=layer.mlp)

            model.model_text.model.forward = partial(
                forward_llama_decoder, self=model.model_text.model
            )
            model.model_text.forward = partial(
                forward_llama,
                self=model.model_text,
                skip_mode=args.skip_mode,
                skip_interval=skip_interval,
                verbose=args.skip_verbose,
                ent_thresh=args.ent_thresh,
                start_layer=args.start_layer,
                end_layer=args.end_layer,
                lm_head=model.model_text.lm_head,
                layers_interval=layers_interval,
                output_intermediate_hidden_states=args.output_intermediate_hidden_states,
            )
        else:

            from pruning.skipping_opt import forward_opt, forward_opt_decoder, forward_opt_layer

            for layer in model.get_llm_layers():
                layer.forward = partial(
                    forward_opt_layer, self=layer, prompt_len=prompt_len
                )  # partial(forward_opt_layer, self=layer)
            model.model_text.model.decoder.forward = partial(
                forward_opt_decoder,
                self=model.model_text.model.decoder,
                prompt_len=prompt_len,
            )
            model.model_text.forward = partial(
                forward_opt,
                self=model.model_text,
                skip_mode=args.skip_mode,
                skip_interval=skip_interval,
                verbose=args.skip_verbose,
                ent_thresh=args.ent_thresh,
                start_layer=args.start_layer,
                end_layer=args.end_layer,
                lm_head=model.model_text.lm_head,
                layers_interval=layers_interval,
                exit_layer=args.exit_layer,
                start_drop_layer=args.start_drop_layer,
                output_intermediate_hidden_states=args.output_intermediate_hidden_states,
                causal_prompt=args.causal_prompt,
            )

    val_evaluator = val_loader.evaluator
    test_evaluator = test_loader.evaluator
    val_task = val_loader.task
    train_task = train_loader.task
    device = accelerator.device

    model, optimizer, train_loader, val_loader, test_loader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_loader, val_loader, test_loader, lr_scheduler
        )
    )

    model = model.to(device)

    test_loader.evaluator = test_evaluator
    val_loader.evaluator = val_evaluator

    test_loader.task = val_task
    val_loader.task = val_task
    train_loader.task = train_task

    accelerator.print("Start training")
    start_time = time.time()

    save_hidden_states = config.get("save_hidden_states", False)
    save_hidden_states_path = os.path.join(
        args.output_dir, args.save_name + "all_hidden_states.pth"
    )
    save_image_feats_path = os.path.join(
        args.output_dir, args.save_name + "image_feats.pth"
    )

    if args.num_beams is not None:
        config["num_beams"] = args.num_beams

    max_length = config.get("max_length", 30)
    print("max length", max_length)
    train_stats = None

    if val_task == "vqa":
        best_metric_key = "Valid/overall"
    else:
        best_metric_key = "Valid/CIDEr"

    for epoch in range(start_epoch, max_epoch):
        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(
                model,
                train_loader,
                optimizer,
                tokenizer,
                epoch,
                warmup_steps,
                device,
                lr_scheduler,
                config,
                accelerator=accelerator,
            )

        if args.evaluate:
            break

        valid_results = evaluation(
            model,
            val_loader,
            tokenizer,
            device,
            config,
            accelerator=accelerator,
            max_length=max_length,
            noise_modality=args.noise_modality,
        )

        if save_hidden_states:
            valid_results = valid_results[0]

        if utils.is_main_process():

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"valid_{k}": v for k, v in valid_results.items()},
                "epoch": epoch,
            }
            save_file = os.path.join(args.output_dir, "log.csv")
            save_logs(save_file, log_stats)
            accelerator.print("save results to:", save_file)

            ## avoid memory issue with accelerator.get_state_dict
            state_dict = accelerator.unwrap_model(model)
            state_dict = state_dict.state_dict()

            state_dict = filter_state(state_dict, exclude_list)  #
            if state_dict is not None:
                for k in state_dict:
                    if state_dict[k].dtype == torch.float16:
                        state_dict[k] = state_dict[k].float()

            save_obj = {
                "model": state_dict,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "config": config,
                "epoch": epoch,
                "best_valid": best_valid,
                "best_epoch": best_epoch,
            }

            if args.save_best:
                valid_score = valid_results[best_metric_key]

                if valid_score > best_valid or epoch == 0:
                    best_valid = valid_score
                    best_epoch = epoch
                    accelerator.print("Save best epoch:", best_epoch)

                    save_obj["best_valid"] = best_valid
                    save_obj["best_epoch"] = best_epoch

                    torch.save(
                        save_obj, os.path.join(args.output_dir, "checkpoint_best.pth")
                    )
            torch.save(save_obj, os.path.join(args.output_dir, "checkpoint_last.pth"))

            if args.save_all:
                torch.save(
                    save_obj, os.path.join(args.output_dir, f"checkpoint_{epoch}.pth")
                )

        dist.barrier()

    if not args.evaluate:
        checkpoint = torch.load(
            os.path.join(args.output_dir, "checkpoint_best.pth"), map_location="cpu"
        )
        state_dict = checkpoint["model"]
        msg = model.module.load_state_dict(state_dict, strict=False)
        msg = filter_msg(msg, exclude_list)
        accelerator.print(
            "load checkpoint for test from %s"
            % os.path.join(args.output_dir, "checkpoint_best.pth")
        )
        accelerator.print(msg)
        print(
            "best_epoch",
            checkpoint["best_epoch"],
            "best_valid",
            checkpoint["best_valid"],
        )
    print("best_epoch", best_epoch, "best_valid", best_valid)

    if accelerator.is_main_process:
        if args.skipping_mode:
            if not args.skip_verbose:
                accelerator.print("*" * 30)
                # model.module.model_text.forward = partial(model.module.model_text.forward, verbose=False)
                accelerator.print("Print generation efficiency stats ...")
                all_flops, all_macs, all_times, all_lens = evaluation_stats(
                    model,
                    test_loader,
                    tokenizer,
                    device,
                    config,
                    accelerator=accelerator,
                    max_length=max_length,
                    only_main=True,
                )
                accelerator.print("*" * 30)

                accelerator.print("*" * 30)
                accelerator.print("Print evaluation/inference efficiency stats ...")
                inf_all_flops, inf_all_macs, inf_all_times, inf_all_lens = (
                    evaluation_stats(
                        model,
                        test_loader,
                        tokenizer,
                        device,
                        config,
                        accelerator=accelerator,
                        max_length=max_length,
                        only_main=True,
                        mode="evaluate",
                    )
                )

                accelerator.print("*" * 30)

    if args.only_compute_stats:
        return

    config["inference_mode"] = args.inference_mode
    config["save_only_image"] = args.save_only_image

    if args.save_hidden_states:
        config["save_hidden_states"] = args.save_hidden_states

    valid_results = evaluation(
        model,
        test_loader,
        tokenizer,
        device,
        config,
        accelerator=accelerator,
        max_length=max_length,
        only_main=False,
        noise_modality=args.noise_modality,
    )

    if utils.is_main_process():

        if args.save_only_image:
            torch.save(
                {
                    "image_feats": valid_results,
                },
                save_image_feats_path,
            )
            print("save image_feats to", save_image_feats_path)
            return
        if config.get(
            "save_hidden_states", False
        ):  # and args.inference_mode != 'generate':

            torch.save(
                {
                    "all_hidden_states": valid_results,
                },
                save_hidden_states_path,
            )
            print("save hidden states to", save_hidden_states_path)

            return

        if args.sparsity_ratio != 0:
            valid_results["sparsity"] = sparsity_ratio
        if args.skipping_mode:

            valid_results["inf_FLOPs"] = inf_all_flops
            valid_results["inf_time"] = inf_all_times
            valid_results["inf_lens"] = inf_all_lens

            valid_results["FLOPs"] = all_flops
            valid_results["time"] = all_times
            valid_results["lens"] = all_lens

        if save_hidden_states:
            valid_results, all_hidden_states = valid_results

        if train_stats is not None:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"valid_{k}": v for k, v in valid_results.items()},
                "epoch": -1,
            }
        else:
            log_stats = {
                **{f"valid_{k}": v for k, v in valid_results.items()},
                "epoch": -2,
            }

        output_log_name = args.output_log_name.replace("/", "_")
        save_logs(os.path.join(args.output_dir, output_log_name), log_stats)
        accelerator.print(
            "save results to:", os.path.join(args.output_dir, output_log_name)
        )

        if save_hidden_states:
            torch.save(
                {
                    "all_hidden_states": all_hidden_states,
                },
                save_hidden_states_path,
            )
            print("save hidden states to", save_hidden_states_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    accelerator.print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/VQA.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output_dir", default="output/vqa")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--text_model", default="facebook/opt-350m")
    parser.add_argument("--vision_model", default="vit_base_patch16_224")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--distributed", default=True, type=bool)

    parser.add_argument("--data_dir", default="/data/mshukor/data")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--save_best", action="store_true")

    parser.add_argument("--image_dir", default="/data/mshukor/data")

    parser.add_argument("--low_cpu", action="store_true")

    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument(
        "--data_json_dir", default="/data/mshukor/data/our_albef_data/data"
    )

    parser.add_argument("--dataset_name", default="coco")

    parser.add_argument("--test_dataset_name", default=None)
    parser.add_argument("--test_data_dir", default=None)

    parser.add_argument("--save_only_image", action="store_true")
    parser.add_argument("--inference_mode", type=str, default="generate")
    parser.add_argument("--save_hidden_states", action="store_true")

    # pruning
    parser.add_argument("--num_calibration_data", default=128, type=int)
    parser.add_argument(
        "--sparsity_ratio", type=float, default=0, help="Sparsity level"
    )
    parser.add_argument(
        "--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"]
    )
    parser.add_argument(
        "--prune_method",
        type=str,
        choices=[
            "given_mask",
            "activation",
            "magnitude",
            "wanda",
            "sparsegpt",
            "ablate_mag_seq",
            "ablate_wanda_seq",
            "ablate_mag_iter",
            "ablate_wanda_iter",
            "search",
            "wanda_block",
            "wandlogit",
        ],
    )
    parser.add_argument(
        "--use_variant",
        action="store_true",
        help="whether to use the wanda variant described in the appendix",
    )
    parser.add_argument("--prune_steps", type=int, default=1, help="pruning steps")
    parser.add_argument("--prune_layer_name", type=str, default="")
    parser.add_argument("--layer_interval", type=int, default=-1)
    parser.add_argument("--prune_save_path", type=str, default=None)
    parser.add_argument("--only_text", action="store_true", help="ignore images")
    parser.add_argument("--only_prompt", action="store_true", help="ignore text")
    parser.add_argument("--with_answers", action="store_true", help="ignore text")
    parser.add_argument("--only_prune", action="store_true", help="only_prune")
    parser.add_argument("--mask_path", type=str, default=None)

    parser.add_argument("--sparsity_mode", type=str, default="")
    parser.add_argument(
        "--ffn_to_sa_sparsity", type=float, default=1, help="ffn_to_sa_sparsity"
    )
    parser.add_argument(
        "--max_value_offset", type=float, default=0, help="ffn_to_sa_sparsity"
    )

    parser.add_argument(
        "--only_compute_stats", action="store_true", help="only_compute_stats"
    )
    parser.add_argument("--ex_sparsity", type=float, default=0.015, help="ex_sparsity")
    parser.add_argument("--per_layer_type", action="store_true", help="per_layer_type")

    parser.add_argument("--test_topk", type=float, default=-1, help="test_topk")

    parser.add_argument(
        "--per_output", action="store_true", help="per output for magnitude"
    )

    parser.add_argument(
        "--min_sparsity_ratio", type=float, default=0, help="min_sparsity_ratio"
    )
    parser.add_argument("--start_layer_first", type=int, default=0)

    # skipping
    parser.add_argument(
        "--skipping_mode", action="store_true", help="whether skip during inference"
    )
    parser.add_argument(
        "--skip_verbose", action="store_true", help="whether skip during inference"
    )
    parser.add_argument("--skip_mode", default="normal", type=str)
    parser.add_argument("--skip_interval", default=60)
    parser.add_argument("--ent_thresh", default=-1, type=float)
    parser.add_argument("--start_layer", default=0, type=int)
    parser.add_argument("--end_layer", default=31, type=int)
    parser.add_argument("--layers_interval", default=0, type=int)
    parser.add_argument("--exit_layer", default=32, type=int)
    parser.add_argument("--start_drop_layer", default=32, type=int)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test_split", default=None)
    parser.add_argument("--test_batch_size", default=None, type=int)
    parser.add_argument("--train_split", default=None)
    parser.add_argument("--val_split", default=None)
    parser.add_argument("--output_log_name", default="log.csv", type=str)
    parser.add_argument("--overwrite_logs", action="store_true")
    parser.add_argument("--noise_modality", action="store_true")
    parser.add_argument("--save_all", action="store_true")
    parser.add_argument("--save_name", default="", type=str)

    parser.add_argument("--output_intermediate_hidden_states", action="store_true")
    parser.add_argument("--keys_to_save", default=None, type=str)

    parser.add_argument("--causal_prompt", action="store_true")

    parser.add_argument("--unimodal_attention_mask", action="store_true")
    parser.add_argument("--stop_unimodal_attention_mask", default=1, type=int)

    parser.add_argument("--num_beams", default=None, type=int)

    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, "result")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, "config.yaml"), "w"))

    main(args, config)
