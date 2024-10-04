# from https://github.com/locuslab/wanda/blob/main/lib/prune.py
import os

import nltk
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .data import get_task_loader, prepare_text_input
from .layerwrapper import WrappedGPT, WrappedGPTLogits


def find_layers(module, layers=[nn.Linear], name="", layer_name=None):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """

    if type(module) in layers:
        if layer_name is not None:
            if any([k in name for k in layer_name]):
                return {name: module}
        else:
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child,
                layers=layers,
                name=name + "." + name1 if name != "" else name1,
                layer_name=layer_name,
            )
        )
    return res


def check_sparsity(layers, accelerator=None, show_layers=False):

    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            ln_count = (W == 0).sum().item()
            ln_params = W.numel()

            sub_count += ln_count
            sub_params += ln_params

            if show_layers:
                if accelerator is not None:
                    accelerator.print(
                        f"layer {i} name: {name} sparsity {float(ln_count)/ln_params:.6f}"
                    )
                else:
                    print(
                        f"layer {i} name: {name} sparsity {float(ln_count)/ln_params:.6f}"
                    )

        if accelerator is not None:
            accelerator.print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")
        else:
            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    return float(count) / total_params


def get_nouns_indices(
    indices,
    tokenizer,
):
    indices_idx = []
    for idx in indices:
        text = tokenizer.decode(idx)
        words = nltk.tokenize.word_tokenize(text)
        pos_tags = nltk.tag.pos_tag(words)
        # Extract nouns based on the POS tags
        nouns = [word for word, pos in pos_tags if pos.startswith("NN")]
        nouns = [word for word in nouns if len(word) > 2]
        noun_indices = [
            tokenizer(n, return_tensors="pt")["input_ids"][0][1].item() for n in nouns
        ]  # skip bos and take the first token if the word is split into 2
        indices_idx.append([i for i in range(len(idx)) if idx[i] in noun_indices])

    return indices_idx


def prepare_multimodal_calibration_input(
    model,
    dataloader,
    device,
    layers,
    batch_size=128,
    tokenizer=None,
    task="coco",
    seqlen=2048,
    hidden_size=4096,
    precision="f32",
    only_prompt=False,
    only_text=False,
    with_answers=False,
    data_idx=-1,
    noise_modality=False,
    get_useful_idx=False,
):

    if noise_modality:
        print("pruning with noise...")

    dtype = next(iter(model.parameters())).dtype

    inps = torch.zeros((batch_size, seqlen, hidden_size), dtype=dtype, device=device)
    attns = torch.zeros((batch_size, 1, seqlen, seqlen), dtype=dtype, device=device)
    posids = torch.zeros((batch_size, seqlen), dtype=dtype, device=device).long()
    inps.requires_grad = False

    cache = {
        "i": 0,
    }

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            bs, l, d = inp.shape
            # assume bs =1 or take the first element in the batch
            inps[cache["i"], :l, :] = inp[0]
            attention_mask = kwargs["attention_mask"]
            bs, l, t, s = attention_mask.shape
            attns[cache["i"], :, :t, :s] = attention_mask[0]

            if "position_ids" in kwargs:
                position_ids = kwargs["position_ids"]
                if position_ids is not None:
                    bs, l = position_ids.shape
                    posids[cache["i"], :l] = position_ids[0]

            cache["i"] += 1

            raise ValueError

    layers[0] = Catcher(layers[0])

    useful_idx = []
    for n, batch in enumerate(dataloader):
        if data_idx >= 0 and n != data_idx:
            continue
        if data_idx >= 0:
            print(
                f"processing {[(k, v) for k, v in batch.items() if 'images' not in k]}"
            )

        image = batch["images"].to(device, non_blocking=True)
        if precision == "f16":
            image = image.half()
        if only_text:
            image = None

        text_input = prepare_text_input(
            batch,
            tokenizer,
            task,
            device,
            dataset="coco",
            only_prompt=only_prompt,
            with_answers=with_answers,
        )
        question_input = text_input

        if get_useful_idx:
            indices = text_input["input_ids"]
            noun_idx = get_nouns_indices(indices, tokenizer)
            # shift to left teacher forcing
            noun_idx = [[i - 1 for i in noun_idx[j]] for j in range(len(noun_idx))]
            useful_idx += noun_idx

        if n >= batch_size:
            break
        try:
            model(
                image=image,
                text=text_input,
                mode="evaluate",
                question=question_input,
                noise_modality=noise_modality,
            )
        except ValueError:
            pass

    layers[0] = layers[0].module
    outs = torch.zeros_like(inps)
    attention_mask = attns
    position_ids = posids  # not used

    if get_useful_idx:
        return inps, outs, attention_mask, position_ids, useful_idx
    else:
        return inps, outs, attention_mask, position_ids


def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    inps.requires_grad = False
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(
        sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1
    )
    W_mask = W_metric <= thres
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_magnitude(
    args,
    model,
    tokenizer,
    layers,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    save_path=None,
    only_prune=False,
    per_output=False,
):
    # layers = model.model_text.model.decoder.layers

    if save_path is not None:
        W_masks = {i: {} for i in range(len(layers))}
        W_metrics = {i: {} for i in range(len(layers))}

        if save_path is not None:
            if args.sparsity_ratio > 0.001:
                mask_path = os.path.join(
                    save_path, f"W_masks_mag_s{args.sparsity_ratio}.pth"
                )
            else:  # save only the metrics when you we don't prune the model
                mask_path = os.path.join(save_path, f"W_mag_metrics.pth")

            if only_prune and os.path.exists(mask_path):
                print(f"Mask exists at {mask_path}, quitting ...")
                return

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = torch.zeros_like(W) == 1
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                if per_output:
                    W_mask = torch.zeros_like(W) == 1
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                    indices = sort_res[1][
                        :, : int(W_metric.shape[1] * args.sparsity_ratio)
                    ]
                    W_mask.scatter_(1, indices, True)
                else:
                    thresh = torch.sort(W_metric.flatten().cuda())[0][
                        int(W.numel() * args.sparsity_ratio)
                    ].cpu()
                    W_mask = W_metric <= thresh

            W[W_mask] = 0

            if save_path is not None:
                W_masks[i].update({name: W_mask.detach().cpu()})
                W_metrics[i].update({name: W_metric.half().detach().cpu()})

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        if args.sparsity_ratio > 0.001:
            torch.save(W_masks, mask_path)
        else:  # save only the metrics when you we don't prune the model
            torch.save(W_metrics, mask_path)


def prune_wanda(
    args,
    model,
    tokenizer,
    layers,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    global_batch_size=128,
    task="coco",
    dataloader=None,
    config=None,
    seqlen=2048,
    hidden_size=4096,
    precision="f32",
    only_activations=False,
    nb_steps=1,
    start_layer=0,
    end_layer=31,
    layer_name=None,
    layer_interval=-1,
    save_path=None,
    only_prompt=False,
    only_text=False,
    with_answers=False,
    is_llama=False,
    only_prune=False,
    verbose=True,
    sparsity_list=None,
    ffn_to_sa_sparsity=1,
    ffn_sparsity_list=None,
    sa_sparsity_list=None,
    accelerator=None,
    data_idx=-1,
    noise_modality=False,
):

    print("loading calibdation data")

    # dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    if dataloader is None:
        dataloader, _ = get_task_loader(**config)

    print("dataset loading complete")

    # prepare input to model layers
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_multimodal_calibration_input(
            model,
            dataloader,
            device,
            layers,
            batch_size=global_batch_size,
            tokenizer=tokenizer,
            task=task,
            seqlen=seqlen,
            hidden_size=hidden_size,
            precision=precision,
            only_prompt=only_prompt,
            only_text=only_text,
            with_answers=with_answers,
            data_idx=data_idx,
            noise_modality=noise_modality,
        )

    sparsity_ratio = (
        args.sparsity_ratio / nb_steps
    )  # (10*args.sparsity_ratio)**(1/nb_steps) / 10
    if layer_interval > 0:
        end_layer = start_layer + layer_interval

    if save_path is not None or data_idx >= 0:
        W_masks = {i: {} for i in range(len(layers))}
        W_metrics = {i: {} for i in range(len(layers))}
        W_acts = {i: {} for i in range(len(layers))}

        if save_path is not None:
            if args.sparsity_ratio > 0.001:
                mask_path = os.path.join(
                    save_path, f"W_masks_s{args.sparsity_ratio}.pth"
                )
            else:  # save only the metrics when you we don't prune the model
                mask_path = os.path.join(save_path, f"W_metrics.pth")

            if only_prune and os.path.exists(mask_path):
                print(f"Mask exists at {mask_path}, quitting ...")
                return

    for j in range(nb_steps):
        if verbose:
            if accelerator is not None:
                accelerator.print(
                    j, f"sparsity_ratio {sparsity_ratio} layer_name: {layer_name}"
                )
            else:
                print(j, f"sparsity_ratio {sparsity_ratio} layer_name: {layer_name}")
        # layers = model.model_text.model.decoder.layers

        inps_, outs_ = inps.clone(), outs.clone()
        for i in range(len(layers)):
            if sparsity_list is not None:
                sparsity_ratio_ = sparsity_list[i]
            else:
                sparsity_ratio_ = sparsity_ratio

            layer = layers[i]
            subset = find_layers(layer, layer_name=layer_name)

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            # first inference to compute the activation
            for j in range(global_batch_size):
                with torch.no_grad():
                    if is_llama:
                        outs_[j] = layer(
                            inps_[j].unsqueeze(0),
                            attention_mask=attention_mask[j].unsqueeze(0),
                            position_ids=position_ids[j].unsqueeze(0),
                        )[0]
                    else:
                        outs_[j] = layer(
                            inps_[j].unsqueeze(0),
                            attention_mask=attention_mask[j].unsqueeze(0),
                        )[0]

            for h in handles:
                h.remove()

            if i >= start_layer and i <= end_layer:
                for name in subset:
                    if ffn_to_sa_sparsity != 1:
                        if any([k in name for k in ["self_attn"]]):
                            if sa_sparsity_list is not None:
                                sparsity_ratio_ = sa_sparsity_list[i]
                            else:
                                sparsity_ratio_ = sparsity_ratio * (
                                    1 - ffn_to_sa_sparsity
                                )  # /0.3333 # B = 0.33 SA + 0.66 FFN
                        else:
                            if ffn_sparsity_list is not None:
                                sparsity_ratio_ = ffn_sparsity_list[i]
                            else:
                                sparsity_ratio_ = (
                                    sparsity_ratio * ffn_to_sa_sparsity
                                )  # /0.6666
                        # sparsity_ratio_ = sparsity_ratio_ * 2
                        sparsity_ratio_ = min(sparsity_ratio_, sparsity_ratio + 0.15)

                    if verbose:
                        if accelerator is not None:
                            accelerator.print(
                                f"pruning layer {i} name {name} sparsity_ratio_{sparsity_ratio_}"
                            )
                        else:
                            print(
                                f"pruning layer {i} name {name} sparsity_ratio_{sparsity_ratio_}"
                            )
                    if only_activations:
                        hout, hin = subset[name].weight.data.shape
                        W_metric = torch.sqrt(
                            wrapped_layers[name].scaler_row.reshape((1, -1))
                        ).repeat(hout, 1)
                    else:
                        W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                            wrapped_layers[name].scaler_row.reshape((1, -1))
                        )

                    if nb_steps > 1:
                        W_metric[W_metric == 0] = W_metric.max()

                    W_mask = (
                        torch.zeros_like(W_metric) == 1
                    )  ## initialize a mask to be all False
                    if prune_n != 0:
                        # structured n:m sparsity
                        for ii in range(W_metric.shape[1]):
                            if ii % prune_m == 0:
                                tmp = W_metric[:, ii : (ii + prune_m)].float()
                                W_mask.scatter_(
                                    1,
                                    ii
                                    + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                                    True,
                                )
                    else:
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)

                        if args.use_variant:
                            # wanda variant
                            tmp_metric = torch.cumsum(sort_res[0], dim=1)
                            sum_before = W_metric.sum(dim=1)

                            alpha = 0.4
                            alpha_hist = [0.0, 0.8]
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before
                            )
                            while (
                                torch.abs(cur_sparsity - sparsity_ratio_) > 0.001
                            ) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                                if cur_sparsity > sparsity_ratio_:
                                    alpha_new = (alpha + alpha_hist[0]) / 2.0
                                    alpha_hist[1] = alpha
                                else:
                                    alpha_new = (alpha + alpha_hist[1]) / 2.0
                                    alpha_hist[0] = alpha

                                alpha = alpha_new
                                W_mask, cur_sparsity = return_given_alpha(
                                    alpha, sort_res, W_metric, tmp_metric, sum_before
                                )
                            print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                        else:
                            # unstructured pruning
                            indices = sort_res[1][
                                :, : int(W_metric.shape[1] * sparsity_ratio_)
                            ]
                            W_mask.scatter_(1, indices, True)

                    subset[name].weight.data[W_mask] = 0  ## set weights to zero

                    if save_path is not None or data_idx >= 0:
                        W_masks[i].update({name: W_mask.detach().cpu()})
                        W_metrics[i].update({name: W_metric.half().detach().cpu()})
                        hout, hin = subset[name].weight.data.shape
                        act = (
                            torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                            .repeat(hout, 1)
                            .half()
                            .detach()
                            .cpu()
                        )
                        W_acts[i].update({name: act})

            # second inference after pruning the layer
            for j in range(global_batch_size):
                with torch.no_grad():
                    if is_llama:
                        outs_[j] = layer(
                            inps_[j].unsqueeze(0),
                            attention_mask=attention_mask[j].unsqueeze(0),
                            position_ids=position_ids[j].unsqueeze(0),
                        )[0]
                    else:
                        outs_[j] = layer(
                            inps_[j].unsqueeze(0),
                            attention_mask=attention_mask[j].unsqueeze(0),
                        )[0]

            # the output is the input to the following layer
            inps_, outs_ = outs_, inps_

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        if args.sparsity_ratio > 0.001:
            torch.save(W_masks, mask_path)
        else:  # save only the metrics when you we don't prune the model
            torch.save(W_metrics, mask_path)

    torch.cuda.empty_cache()
    if data_idx >= 0:
        return W_metrics


def prune_wandlogit(
    args,
    model,
    tokenizer,
    layers,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    global_batch_size=128,
    task="coco",
    dataloader=None,
    config=None,
    seqlen=2048,
    hidden_size=4096,
    precision="f32",
    only_activations=False,
    nb_steps=1,
    start_layer=0,
    end_layer=31,
    layer_name=None,
    layer_interval=-1,
    save_path=None,
    only_prompt=False,
    only_text=False,
    with_answers=False,
    is_llama=False,
    only_prune=False,
    verbose=True,
    sparsity_list=None,
    ffn_to_sa_sparsity=1,
    ffn_sparsity_list=None,
    sa_sparsity_list=None,
    accelerator=None,
    data_idx=-1,
    noise_modality=False,
    layer_norm=None,
    lm_head=None,
):

    # use_cache = model.model_text.config.use_cache
    # model.model_text.config.use_cache = False

    print("loading calibdation data")

    # dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    if dataloader is None:
        dataloader, _ = get_task_loader(**config)

    print("dataset loading complete")

    # prepare input to model layers
    with torch.no_grad():
        (
            inps,
            outs,
            attention_mask,
            position_ids,
            useful_indices,
        ) = prepare_multimodal_calibration_input(
            model,
            dataloader,
            device,
            layers,
            batch_size=global_batch_size,
            tokenizer=tokenizer,
            task=task,
            seqlen=seqlen,
            hidden_size=hidden_size,
            precision=precision,
            only_prompt=only_prompt,
            only_text=only_text,
            with_answers=with_answers,
            data_idx=data_idx,
            noise_modality=noise_modality,
            get_useful_idx=True,
        )

    sparsity_ratio = (
        args.sparsity_ratio / nb_steps
    )  # (10*args.sparsity_ratio)**(1/nb_steps) / 10
    if layer_interval > 0:
        end_layer = start_layer + layer_interval

    if save_path is not None or data_idx >= 0:
        W_masks = {i: {} for i in range(len(layers))}
        W_metrics = {i: {} for i in range(len(layers))}
        W_acts = {i: {} for i in range(len(layers))}

        if save_path is not None:
            if args.sparsity_ratio > 0.001:
                mask_path = os.path.join(
                    save_path, f"W_masks_s{args.sparsity_ratio}.pth"
                )
            else:  # save only the metrics when you we don't prune the model
                mask_path = os.path.join(save_path, f"W_metrics.pth")

            if only_prune and os.path.exists(mask_path):
                print(f"Mask exists at {mask_path}, quitting ...")
                return

    for j in range(nb_steps):
        if verbose:
            if accelerator is not None:
                accelerator.print(
                    j, f"sparsity_ratio {sparsity_ratio} layer_name: {layer_name}"
                )
            else:
                print(j, f"sparsity_ratio {sparsity_ratio} layer_name: {layer_name}")
        # layers = model.model_text.model.decoder.layers

        inps_, outs_ = inps.clone(), outs.clone()
        for i in range(len(layers)):
            if sparsity_list is not None:
                sparsity_ratio_ = sparsity_list[i]
            else:
                sparsity_ratio_ = sparsity_ratio

            layer = layers[i]
            subset = find_layers(layer, layer_name=layer_name)

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPTLogits(
                    subset[name],
                    layer_norm=layer_norm,
                    lm_head=lm_head,
                    indices=useful_indices,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            # first inference to compute the activation
            for j in range(global_batch_size):
                with torch.no_grad():
                    if is_llama:
                        outs_[j] = layer(
                            inps_[j].unsqueeze(0),
                            attention_mask=attention_mask[j].unsqueeze(0),
                            position_ids=position_ids[j].unsqueeze(0),
                        )[0]
                    else:
                        outs_[j] = layer(
                            inps_[j].unsqueeze(0),
                            attention_mask=attention_mask[j].unsqueeze(0),
                        )[0]

            for h in handles:
                h.remove()

            if i >= start_layer and i <= end_layer:
                for name in subset:
                    if ffn_to_sa_sparsity != 1:
                        if any([k in name for k in ["self_attn"]]):
                            if sa_sparsity_list is not None:
                                sparsity_ratio_ = sa_sparsity_list[i]
                            else:
                                sparsity_ratio_ = sparsity_ratio * (
                                    1 - ffn_to_sa_sparsity
                                )  # /0.3333 # B = 0.33 SA + 0.66 FFN
                        else:
                            if ffn_sparsity_list is not None:
                                sparsity_ratio_ = ffn_sparsity_list[i]
                            else:
                                sparsity_ratio_ = (
                                    sparsity_ratio * ffn_to_sa_sparsity
                                )  # /0.6666
                        # sparsity_ratio_ = sparsity_ratio_ * 2
                        sparsity_ratio_ = min(sparsity_ratio_, sparsity_ratio + 0.15)

                    if verbose:
                        if accelerator is not None:
                            accelerator.print(
                                f"pruning layer {i} name {name} sparsity_ratio_{sparsity_ratio_}"
                            )
                        else:
                            print(
                                f"pruning layer {i} name {name} sparsity_ratio_{sparsity_ratio_}"
                            )
                    if only_activations:
                        hout, hin = subset[name].weight.data.shape
                        W_metric = torch.sqrt(
                            wrapped_layers[name].scaler_row.reshape((1, -1))
                        ).repeat(hout, 1)
                    else:
                        W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                            wrapped_layers[name].scaler_row.reshape((1, -1))
                        )

                    if nb_steps > 1:
                        W_metric[W_metric == 0] = W_metric.max()

                    W_mask = (
                        torch.zeros_like(W_metric) == 1
                    )  ## initialize a mask to be all False
                    if prune_n != 0:
                        # structured n:m sparsity
                        for ii in range(W_metric.shape[1]):
                            if ii % prune_m == 0:
                                tmp = W_metric[:, ii : (ii + prune_m)].float()
                                W_mask.scatter_(
                                    1,
                                    ii
                                    + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                                    True,
                                )
                    else:
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)

                        if args.use_variant:
                            # wanda variant
                            tmp_metric = torch.cumsum(sort_res[0], dim=1)
                            sum_before = W_metric.sum(dim=1)

                            alpha = 0.4
                            alpha_hist = [0.0, 0.8]
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before
                            )
                            while (
                                torch.abs(cur_sparsity - sparsity_ratio_) > 0.001
                            ) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                                if cur_sparsity > sparsity_ratio_:
                                    alpha_new = (alpha + alpha_hist[0]) / 2.0
                                    alpha_hist[1] = alpha
                                else:
                                    alpha_new = (alpha + alpha_hist[1]) / 2.0
                                    alpha_hist[0] = alpha

                                alpha = alpha_new
                                W_mask, cur_sparsity = return_given_alpha(
                                    alpha, sort_res, W_metric, tmp_metric, sum_before
                                )
                            print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                        else:
                            # unstructured pruning
                            indices = sort_res[1][
                                :, : int(W_metric.shape[1] * sparsity_ratio_)
                            ]
                            W_mask.scatter_(1, indices, True)

                    subset[name].weight.data[W_mask] = 0  ## set weights to zero

                    if save_path is not None or data_idx >= 0:
                        W_masks[i].update({name: W_mask.detach().cpu()})
                        W_metrics[i].update({name: W_metric.half().detach().cpu()})
                        hout, hin = subset[name].weight.data.shape
                        act = (
                            torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                            .repeat(hout, 1)
                            .half()
                            .detach()
                            .cpu()
                        )
                        W_acts[i].update({name: act})

            # second inference after pruning the layer
            for j in range(global_batch_size):
                with torch.no_grad():
                    if is_llama:
                        outs_[j] = layer(
                            inps_[j].unsqueeze(0),
                            attention_mask=attention_mask[j].unsqueeze(0),
                            position_ids=position_ids[j].unsqueeze(0),
                        )[0]
                    else:
                        outs_[j] = layer(
                            inps_[j].unsqueeze(0),
                            attention_mask=attention_mask[j].unsqueeze(0),
                        )[0]

            # the output is the input to the following layer
            inps_, outs_ = outs_, inps_

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        if args.sparsity_ratio > 0.001:
            torch.save(W_masks, mask_path)
        else:  # save only the metrics when you we don't prune the model
            torch.save(W_metrics, mask_path)
        # torch.save(W_acts, os.path.join(save_path, "W_acts.pth"))

    # model.model_text.config.use_cache = use_cache
    torch.cuda.empty_cache()
    if data_idx >= 0:
        return W_metrics


layer_names = [
    "fc1",
    "fc2",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.q_proj",
    "self_attn.out_proj",
]


def compute_ex_mask(
    mask_1,
    mask_2=None,
    s=0.015,
    layer_names=[
        "fc1",
        "fc2",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.q_proj",
        "self_attn.out_proj",
    ],
    startl=0,
    mode="ex",
):
    mask = {i: {} for i in range(len(mask_1))}
    for i in tqdm(range(len(mask_1))):
        if i >= startl:
            sub_count = 0
            sub_params = 0
            for layer_name in layer_names:
                matrix1 = ~mask_1[i][
                    layer_name
                ].numpy()  # so 1 for valid and 0 for masked weight

                if mask_2 is None:  # random mask

                    num_true_values_to_change = int(
                        s * matrix1.size
                    )  # 3% of the weights put to zero
                    true_indices = np.argwhere(matrix1)
                    indices_to_change = np.random.choice(
                        len(true_indices), size=num_true_values_to_change, replace=False
                    )
                    matrix2 = np.copy(matrix1)
                    matrix2[tuple(true_indices[indices_to_change].T)] = False
                    mask[i][layer_name] = torch.tensor(~matrix2)
                else:
                    matrix2 = ~mask_2[i][layer_name].numpy()

                    intersection = np.logical_and(matrix1, matrix2)
                    if mode == "intersection":
                        modified_matrix1 = intersection
                    elif mode == "union":
                        modified_matrix1 = np.logical_or(matrix1, matrix2)
                    else:
                        num_true_values_to_replace = min(
                            int(s * matrix1.size), np.sum(intersection)
                        )  # int(0.01 * s * np.sum(intersection))
                        intersection_indices = np.argwhere(intersection)
                        indices_to_replace = np.random.choice(
                            len(intersection_indices),
                            size=num_true_values_to_replace,
                            replace=False,
                        )
                        modified_matrix1 = np.copy(matrix1)
                        modified_matrix1[
                            tuple(intersection_indices[indices_to_replace].T)
                        ] = False

                    mask[i][layer_name] = torch.tensor(~modified_matrix1)

        else:
            print(f"skip {i} {layer_name}")
    return mask


def prune_given_mask(
    layers,
    start_layer=0,
    end_layer=31,
    layer_name=None,
    layer_interval=-1,
    mask_path=None,
    ex_sparsity=0.015,
):

    # use_cache = model.model_text.config.use_cache
    # model.model_text.config.use_cache = False
    if "!!!" in mask_path:  # ablate
        mask_path1, mask_path2 = mask_path.split("!!!")
        W_masks1 = torch.load(mask_path1)
        if "random" in mask_path2:
            W_masks2 = None
        else:
            W_masks2 = torch.load(mask_path2)

        W_masks = compute_ex_mask(W_masks1, W_masks2, s=ex_sparsity)
    elif "???" in mask_path:
        print("compute intersection mask")
        mask_path1, mask_path2 = mask_path.split("???")
        W_masks1 = torch.load(mask_path1)
        if "random" in mask_path2:
            W_masks2 = None
        else:
            W_masks2 = torch.load(mask_path2)

        W_masks = compute_ex_mask(
            W_masks1, W_masks2, s=ex_sparsity, mode="intersection"
        )
    elif "!?!" in mask_path:
        print("compute union mask")
        mask_path1, mask_path2 = mask_path.split("!?!")
        W_masks1 = torch.load(mask_path1)
        if "random" in mask_path2:
            W_masks2 = None
        else:
            W_masks2 = torch.load(mask_path2)

        W_masks = compute_ex_mask(W_masks1, W_masks2, s=ex_sparsity, mode="union")
    elif "random" in mask_path:

        ### if random mask is not already saved
        W_masks = {i: {} for i in range(end_layer - start_layer + 1)}

        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer, layer_name=layer_name)
            if i >= start_layer and i <= end_layer:
                for name in subset:
                    W_mask = np.ones_like(subset[name].weight.cpu().numpy(), dtype=bool)
                    num_false_values_to_change = int(
                        (1 - ex_sparsity) * W_mask.size
                    )  # 3% of the weights put to zero
                    true_indices = np.argwhere(W_mask)
                    indices_to_change = np.random.choice(
                        len(true_indices),
                        size=num_false_values_to_change,
                        replace=False,
                    )
                    W_mask[tuple(true_indices[indices_to_change].T)] = False
                    W_masks[i][name] = torch.tensor(W_mask)
        ## save the mask

    else:
        W_masks = torch.load(mask_path)

    if layer_interval > 0:
        end_layer = start_layer + layer_interval

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer, layer_name=layer_name)

        if i >= start_layer and i <= end_layer:
            for name in subset:

                W_mask = W_masks[i][name].to(subset[name].weight.device)

                subset[name].weight.data[W_mask] = 0  ## set weights to zero

    # model.model_text.config.use_cache = use_cache
    torch.cuda.empty_cache()


def top_k_percent_across_matrices(matrices, k):

    matrices = [m.detach().half().cpu().numpy() for m in matrices]
    # Determine the common shape for the result matrix

    common_shape = np.max([matrix.shape for matrix in matrices], axis=0)

    # Resize each matrix to the common shape
    resized_matrices = [np.resize(matrix, common_shape) for matrix in matrices]

    # Flatten and concatenate the resized matrices
    flat_tensor = np.concatenate([matrix.flatten() for matrix in resized_matrices])
    flat_tensor.sort()

    # Calculate the threshold value for the top k percent
    threshold = np.percentile(flat_tensor, 100 - k)

    return threshold
