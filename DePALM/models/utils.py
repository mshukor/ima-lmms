import csv

import numpy as np
import torch

exclude_list = ["model_text", "transformer", "model_vision"]


def filter_msg(msg, exclude_list):
    new_msg = []
    if len(msg) > 1:
        for k in msg[0]:  # missing
            if not any([e in k for e in exclude_list]) or "adapter" in k:
                new_msg.append(k)
        return new_msg


def filter_state(state, exclude_list):
    import collections

    new_tmp = collections.OrderedDict()
    for k, v in state.items():
        if not any([e in k for e in exclude_list]) or "adapter" in k:
            new_tmp[k] = state[k]

    return new_tmp


def freeze_whole_model(model):
    for n, p in model.named_parameters():
        p.requires_grad = False


def unfreeze_parameters(model, config, accelerator=None):
    # targets = '*.proj_*|*_proj*|*itm_head*|*queue*|*adapter*|*temp*|*.cls.*'

    targets = ["prompt", "gate"]  # lm_head 'cross_modal_module'
    exceptions = ["gate_proj"]

    if config.get("block_config", False):
        targets = targets + ["cross_modal_module"]

    if not config.get("freeze_connector", False):
        targets = targets + ["connector"]

    if config.get("unfreeze_text_layer_norm", False):
        targets = targets + ["self_attn_layer_norm", "final_layer_norm"]

    if config.get("unfreeze_vision_layer_norm", False):
        targets = targets + ["norm", "norm1", "norm2"]

    if config.get("unfreeze_text_model", False):
        targets = targets + ["model_text"]

    if config.get("unfreeze_vision_model", False):
        targets = targets + ["model_vision"]

    if config.get("use_adapters", False):
        targets = targets + ["adapter"]

    if config.get("use_lora", False):
        targets = targets + ["lora"]

    print("unfreeze targets:", targets)
    for n, p in model.named_parameters():
        if any(t in n for t in targets):
            # if re.fullmatch(targets, n):
            if not any(t in n for t in exceptions):
                p.requires_grad = True
                if accelerator is not None:
                    accelerator.print(f"{n} is trainable...")
                else:
                    print(f"{n} is trainable...")


def print_trainable_params_percentage(model, accelerator=None):

    orig_param_size = sum(p.numel() for p in model.parameters())

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable_size = count_parameters(model)

    percentage = trainable_size / orig_param_size * 100

    if accelerator is not None:
        accelerator.print(
            f"Trainable param percentage: {percentage:.2f}% ({trainable_size}/{orig_param_size})"
        )
    else:
        print(
            f"Trainable param percentage: {percentage:.2f}% ({trainable_size}/{orig_param_size})"
        )

    return percentage


def shift_right(input_ids, decoder_start_token_id=2, pad_token_id=None):

    # shift inputs to the right
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id

    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    assert torch.all(
        shifted_input_ids >= 0
    ).item(), "Verify that `shifted_input_ids` has only positive values"

    return shifted_input_ids


def remove_letters(input_string):
    # Use a list comprehension to create a new string without letters
    result_string = "".join(char for char in input_string if not char.isalpha())
    return result_string


def save_logs(csv_file, logs, overwrite=False):

    # Open the CSV file in append mode
    for k, v in logs.items():
        if isinstance(v, float):
            v = round(v, 4)
            v = v * 100 if v < 2 else v

    # if overwrite:
    #     key = 'a'
    # else:
    #     key = 'a'
    with open(csv_file, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=logs.keys())

        # If the file is empty, write the header
        if file.tell() == 0:
            writer.writeheader()

        # Write the new data to the CSV file
        writer.writerow(logs)


def get_gate(model):
    gates = {}
    for name, param in model.named_parameters():
        if "gate" in name and "gate_proj" not in name:
            try:
                gates[name] = param.squeeze().item()
            except:
                gates[name] = [g.item() for g in param.squeeze()]
    return gates


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    )
    return torch.index_select(x, dim, order_index.to(x.device))


#### For ICL inspired from openflamingo


def get_random_indices(
    num_samples, query_set_size, full_dataset, seed, query_dataset=None
):
    if query_dataset is None and num_samples + query_set_size > len(full_dataset):
        raise ValueError(
            f"num_samples {num_samples} + num_shots {query_set_size}  must be less than {len(full_dataset)}"
        )

    # get a random subset of the dataset
    np.random.seed(seed)
    if query_dataset is not None:
        random_indices = np.random.choice(len(full_dataset), num_samples, replace=False)
        query_indices = np.random.choice(
            len(query_dataset), query_set_size, replace=False
        )

        return (random_indices, query_indices)

    else:
        random_indices = np.random.choice(
            len(full_dataset), num_samples + query_set_size, replace=False
        )
        return random_indices


def prepare_eval_samples_and_dataset(
    full_dataset,
    random_indices,
    query_set_size,
    query_indices=None,
    query_dataset=None,
    mode=None,
):
    # get in context samples

    in_context_samples = [full_dataset[i] for i in random_indices[:query_set_size]]

    eval_dataset = torch.utils.data.Subset(
        full_dataset, random_indices[query_set_size:]
    )
    return in_context_samples, eval_dataset
