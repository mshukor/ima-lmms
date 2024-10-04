from dataset.caption import get_loader
from dataset.llava import get_loader as get_loader_llava
from dataset.vqa import get_loader as get_loader_vqa

# from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


# prepare iput upt to the first player then  for each layer: then for each sample do as wanda put with only embed/att or only hidden states


# def prepare_layer_input_opt(
#     model,
#     input_ids: torch.LongTensor = None,
#     attention_mask: Optional[torch.Tensor] = None,
#     head_mask: Optional[torch.Tensor] = None,
#     past_key_values: Optional[List[torch.FloatTensor]] = None,
#     inputs_embeds: Optional[torch.FloatTensor] = None,
#     use_cache: Optional[bool] = None,
#     output_attentions: Optional[bool] = None,
#     output_hidden_states: Optional[bool] = None,
#     return_dict: Optional[bool] = None,
# ):
#     output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
#     output_hidden_states = (
#         output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
#     )
#     use_cache = use_cache if use_cache is not None else model.config.use_cache

#     return_dict = return_dict if return_dict is not None else model.config.use_return_dict

#     # retrieve input_ids and inputs_embeds
#     if input_ids is not None and inputs_embeds is not None:
#         raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
#     elif input_ids is not None:
#         input_shape = input_ids.size()
#         input_ids = input_ids.view(-1, input_shape[-1])
#     elif inputs_embeds is not None:
#         input_shape = inputs_embeds.size()[:-1]
#     else:
#         raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

#     if inputs_embeds is None:
#         inputs_embeds = model.embed_tokens(input_ids)

#     batch_size, seq_length = input_shape
#     past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
#     # required mask seq length can be calculated via length of past
#     mask_seq_length = past_key_values_length + seq_length

#     # embed positions
#     if attention_mask is None:
#         attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
#     elif attention_mask.shape[1] != mask_seq_length:
#         raise ValueError(
#             f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
#             f"{mask_seq_length} (sum of the lengths of current and past inputs)"
#         )
#     causal_attention_mask = _prepare_4d_causal_attention_mask(
#         attention_mask, input_shape, inputs_embeds, past_key_values_length
#     )
#     pos_embeds = model.embed_positions(attention_mask, past_key_values_length)

#     if model.project_in is not None:
#         inputs_embeds = model.project_in(inputs_embeds)

#     hidden_states = inputs_embeds + pos_embeds

#     if model.gradient_checkpointing and model.training:
#         if use_cache:
#             print(
#                 "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
#             )
#             use_cache = False


#     # check if head_mask has a correct number of layers specified if desired
#     for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
#         if attn_mask is not None:
#             if attn_mask.size()[0] != (len(model.layers)):
#                 raise ValueError(
#                     f"The `{mask_name}` should be specified for {len(model.layers)} layers, but it is for"
#                     f" {head_mask.size()[0]}."
#                 )


#     return hidden_states, causal_attention_mask


def prepare_text_input(
    batch,
    tokenizer,
    task,
    device,
    dataset="coco",
    add_context=False,
    instruction="",
    only_prompt=False,
    with_answers=False,
):

    if only_prompt:
        text = ["" for q in batch["sent"]]
        text_input = tokenizer(text, padding="longest", return_tensors="pt").to(device)
        return text_input

    if with_answers:
        if task == "vqa":
            if dataset == "llava":
                text = [
                    f"Human:{q} Assistant:{a.replace('[SEP]','')}"
                    for q, a in zip(batch["sent"], batch["answers"])
                ]
            else:
                text = [
                    f"Question:{q} Answer:{a.replace('[SEP]','')}"
                    for q, a in zip(batch["sent"], batch["answers"])
                ]
        elif task == "llava":
            text = [f"{q}" for q in batch["sent"]]
            if add_context:
                text = [f"{batch['context'][i]} {text[i]}" for i in range(len(text))]
        else:
            text = [instruction + t for t in batch["sent"]]
    else:
        if task == "vqa":
            if dataset == "llava":
                text = [f"Human:{q} Assistant:" for q in batch["sent"]]
            else:
                text = [f"Question:{q} Answer:" for q in batch["sent"]]
        elif task == "llava":
            text = [f"{q}" for q in batch["sent"]]
            if add_context:
                text = [f"{batch['context'][i]} {text[i]}" for i in range(len(text))]
        else:
            text = [instruction for q in range(len(batch["img_id"]))]

    text_input = tokenizer(text, padding="longest", return_tensors="pt").to(device)

    return text_input


def get_task_loader(
    dataset_name,
    test_dataset_name=None,
    image_size=224,
    data_json_dir=None,
    image_dir=None,
    batch_size_train=2,
    batch_size_test=4,
    num_workers=2,
    train_topk=-1,
    valid_topk=-1,
    global_rank=0,
    num_tasks=1,
    config_dir=None,
    data_dir=None,
    test_data_dir=None,
    train_split="train",
    val_split="val",
    test_split="test",
    **kwargs,
):

    distributed = False
    if dataset_name == "vqav2":
        get_loader_ = get_loader_vqa
    elif dataset_name == "llava":
        get_loader_ = get_loader_llava
    else:
        get_loader_ = get_loader

    test_dataset_name = (
        test_dataset_name if test_dataset_name is not None else dataset_name
    )

    if test_dataset_name == "vqav2":
        get_loader_eval = get_loader_vqa
    else:
        get_loader_eval = get_loader

    raw_label = False
    use_data_augmentation = True

    # if dataset_name == 'CC3M':
    #     train_loader, train_dataset = get_loader_(
    #         split='CC3M', mode='train', batch_size=batch_size_train,
    #         distributed=distributed,
    #         workers=num_workers,
    #         topk=train_topk,
    #         data_dir=image_dir,
    #         config_dir=config_dir,
    #         local_rank=global_rank, world_size=num_tasks, verbose=True,
    #         image_size=image_size, use_data_augmentation=use_data_augmentation,
    #         data_json_dir=data_json_dir,
    #     )
    # elif dataset_name == 'vqav2':

    #     train_loader, train_dataset  = get_loader_(
    #         split=train_split, mode='train', batch_size=batch_size_train,
    #         distributed=distributed,
    #         workers=num_workers,
    #         topk=train_topk,
    #         data_dir=data_dir,
    #         local_rank=global_rank, world_size=num_tasks, verbose=True,
    #         image_size=image_size, use_data_augmentation=use_data_augmentation, raw_label=raw_label,
    #     )
    # else:
    train_loader, train_dataset = get_loader_(
        split="train",
        mode="train",
        batch_size=batch_size_train,
        distributed=distributed,
        workers=num_workers,
        topk=train_topk,
        data_dir=data_dir,
        local_rank=global_rank,
        world_size=num_tasks,
        verbose=True,
        image_size=image_size,
        use_data_augmentation=use_data_augmentation,
    )

    # # if gpu == 0:
    # print('# len train loader:', len(train_loader))

    # test_data_dir = test_data_dir if test_data_dir is not None else data_dir
    # print(f'Building val loader')
    # val_loader, val_dataset  = get_loader_eval(
    #     split=val_split, mode='val', batch_size=batch_size_test,
    #     distributed=False,
    #     workers=4,
    #     topk=valid_topk,data_dir=test_data_dir,
    #     local_rank=global_rank, world_size=num_tasks, verbose=True,
    #     image_size=image_size, use_data_augmentation=use_data_augmentation,
    # )
    # print('# len val loader:', len(val_loader))

    print(f"Building test loader")
    test_loader, test_dataset = get_loader_eval(
        split=test_split,
        mode="val",
        batch_size=batch_size_test,
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

    return (
        test_loader,
        test_dataset,
        train_loader,
        train_dataset,
    )
