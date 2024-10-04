import random
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)


def decode_text(
    h=None,
    lm_head=None,
    processor=None,
    k=1,
    get_text=False,
):
    logits = lm_head(h)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    index = probs.topk(dim=-1, k=k)[-1]
    # index = index.squeeze(-1)
    if k == 1 and processor is not None and get_text:
        text = processor.batch_decode(index, skip_special_tokens=True)[0]
    else:
        text = ""

    return text, probs, index


def forward_opt_decoder(
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    verbose=False,
    skip_interval=2,
    skip_mode="normal",
    self=None,
    ent_thresh=100,
    start_layer=0,
    end_layer=31,
    lm_head=None,
    layers_interval=[],
    exit_layer=32,
    start_drop_layer=32,
    output_intermediate_hidden_states=False,
    causal_prompt=False,
    prompt_len=10,
) -> Union[Tuple, BaseModelOutputWithPast]:
    r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
            provide it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
            cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
            that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
            all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
    """
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_length = input_shape
    past_key_values_length = (
        past_key_values[0][0].shape[2] if past_key_values is not None else 0
    )
    # required mask seq length can be calculated via length of past
    mask_seq_length = past_key_values_length + seq_length

    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            batch_size, mask_seq_length, device=inputs_embeds.device
        )
    elif attention_mask.shape[1] != mask_seq_length:
        raise ValueError(
            f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
            f"{mask_seq_length} (sum of the lengths of current and past inputs)"
        )
    causal_attention_mask = _prepare_4d_causal_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_key_values_length
    )

    # if causal_prompt:
    #     print(causal_attention_mask, 'before', causal_attention_mask.shape)
    #     causal_attention_mask = torch.triu(causal_attention_mask, diagonal=0)# bs, 1, ql, sl
    #     print(causal_attention_mask, 'after', causal_attention_mask.shape)

    pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

    if self.project_in is not None:
        inputs_embeds = self.project_in(inputs_embeds)

    hidden_states = inputs_embeds + pos_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            print(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    # check if head_mask has a correct number of layers specified if desired
    for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
        if attn_mask is not None:
            if attn_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

    layer_outputs_prev = None

    num_skips = 0

    # skip_interval = skip_interval ## list of 3/4 steps [1, 2, 3, 4] + add parrallel block
    if isinstance(skip_interval, list):
        len_skip_int = len(skip_interval)
        step = len(self.layers) // len_skip_int

    # layers_interval = [0, 2, 4, 6, 8, 10, ] for skip_interval = 2 [2, 2, 2, ] block[i] + block[i+skip_interval]

    previous_token_index = None
    exited = False

    if output_intermediate_hidden_states:
        all_hidden_states += ((hidden_states, (torch.zeros_like(hidden_states),) * 5),)
    elif output_hidden_states:
        all_hidden_states += (hidden_states,)

    for idx, decoder_layer in enumerate(self.layers):
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        # print(idx, len(past_key_values))
        # if 'parrallel_block' in skip_mode and layers_interval is not None and idx not in layers_interval:
        #     continue

        # if 'parrallel_block' in skip_mode and (idx+1) % skip_interval_:
        #     continue

        if isinstance(skip_interval, list):
            skip_interval_ = skip_interval[min(idx // step, len_skip_int - 1)]
        else:
            skip_interval_ = skip_interval

        if "random" in skip_mode:
            skip_interval_ = random.choice(list(range(2, skip_interval_ + 1)))

        if verbose:
            print(idx, skip_interval_, skip_interval)

        if self.training:
            dropout_probability = torch.rand([])
            if dropout_probability < self.layerdrop:
                continue

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if ent_thresh > 0 or "token_index" in skip_mode:

            text, probs, best_token_index = decode_text(
                h=hidden_states, lm_head=lm_head, processor=None, k=1
            )
            ent = (
                torch.distributions.Categorical(probs[0][-1]).entropy().item()
            )  # works with bs =1

            best_token_index = best_token_index[0][-1][
                0
            ]  # works with batch size =1 ignore beam search, last token, top 1

            if previous_token_index is not None:
                if best_token_index == previous_token_index:
                    skip_interval_ = max(2, skip_interval_ - 1)
                # else:
                #     skip_interval_ += 1
        else:
            best_token_index = 0
            previous_token_index = 1
            ent = 100

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_attention_mask,
                head_mask[idx] if head_mask is not None else None,
                None,
                output_attentions,
                use_cache,
            )
        else:
            bs, l, d = hidden_states.shape

            mode = "normal"

            min_skip_at_len = 0
            if "context" in skip_mode:
                if "only" in skip_mode:
                    min_skip_at_len = 5  # at least 5 image tokens 10 is fine
                skip_at_len = 2100
            else:
                skip_at_len = 1

            if "exit" in skip_mode and not exited:
                skip_at_len = 0
                if idx > exit_layer:
                    if l > 1:
                        if "prompt" in skip_mode:
                            hidden_states = hidden_states[:, 10::]
                            causal_attention_mask = causal_attention_mask[
                                :, :, 10:, 10:
                            ]
                        else:
                            hidden_states = hidden_states[:, -1::]
                            causal_attention_mask = causal_attention_mask[
                                :, :, -1:, -1:
                            ]

                    if l == 1:
                        if "prompt" in skip_mode:
                            causal_attention_mask = causal_attention_mask[:, :, :, 10:]
                        else:
                            causal_attention_mask = causal_attention_mask[:, :, :, -1:]

                    exited = True
                    # print(hidden_states.shape, idx, 'exit', causal_attention_mask.shape)

            if (
                (l <= skip_at_len and l > min_skip_at_len)
                and layer_outputs_prev is not None
                and idx < (end_layer)
                and idx > (start_layer)
            ):  # len(self.layers)-1

                if idx % skip_interval_ == 0 or ent < ent_thresh:

                    if "skip_attn" in skip_mode:
                        mode = "skip_attn"
                        num_skips += 1
                    elif "parrallel_attn" in skip_mode:
                        mode = "parrallel"
                        num_skips += 1
                    elif "parrallel_block" in skip_mode:
                        mode = "parrallelblock"
                        num_skips += 1
                    elif "skip_ffn" in skip_mode:
                        mode = "skip_ffn"
                        num_skips += 1

            if "prompt" in skip_mode:
                mode += "_prompt"

            if (
                (l <= skip_at_len and l > min_skip_at_len)
                and layer_outputs_prev is not None
                and idx < (end_layer)
                and idx > (start_layer)
                and ("skip_block" in skip_mode or "parrallel_block" in skip_mode)
            ):

                if "parrallel_block" in skip_mode and (
                    idx in layers_interval or (idx - 1) in layers_interval
                ):
                    if (idx - 1) in layers_interval:
                        layer_outputs = layer_outputs_prev
                        num_skips += 1
                        if verbose:
                            print(idx, "skip")
                    elif idx in layers_interval:
                        layer_outputs_js = []
                        for j in range(idx, idx + 2):
                            layer_outputs1 = self.layers[j](
                                hidden_states,
                                attention_mask=causal_attention_mask,
                                layer_head_mask=(
                                    head_mask[idx] if head_mask is not None else None
                                ),
                                past_key_value=past_key_value,
                                output_attentions=output_attentions,
                                use_cache=use_cache,
                                att_mode=mode,
                            )
                            layer_outputs_js.append(layer_outputs1)

                        layer_outputs = list(layer_outputs_js[-1])
                        layer_outputs[0] = sum(
                            [lo[0] for lo in layer_outputs_js]
                        ) / len(layer_outputs_js)

                        if use_cache:
                            if output_attentions:
                                kv_idx = 2
                            else:
                                kv_idx = 1
                            k_ = sum([lo[kv_idx][0] for lo in layer_outputs_js]) / len(
                                layer_outputs_js
                            )
                            v_ = sum([lo[kv_idx][1] for lo in layer_outputs_js]) / len(
                                layer_outputs_js
                            )

                            layer_outputs[kv_idx] = (k_, v_)

                        layer_outputs = tuple(layer_outputs)
                        # print(idx, "parralel", len(layer_outputs_js))

                elif idx % skip_interval_ == 0 or ent < ent_thresh:
                    if "prompt" in skip_mode and hidden_states.shape[1] > prompt_len:
                        hidden_states_ = hidden_states[:, prompt_len:, :]

                        causal_attention_mask_ = causal_attention_mask[
                            :, :, prompt_len:, prompt_len:
                        ]

                        layer_outputs_ = decoder_layer(
                            hidden_states_,
                            attention_mask=causal_attention_mask_,
                            layer_head_mask=(
                                head_mask[idx] if head_mask is not None else None
                            ),
                            past_key_value=past_key_value,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            att_mode=mode,
                        )
                        hidden_states_ = layer_outputs_[0]
                        hidden_states = torch.cat(
                            (hidden_states[:, :prompt_len, :], hidden_states_), dim=1
                        )

                        layer_outputs = list(layer_outputs_prev)
                        layer_outputs[0] = hidden_states
                        layer_outputs = tuple(layer_outputs)

                    else:
                        layer_outputs = layer_outputs_prev

                    num_skips += 1
                    if verbose:
                        print(idx, "skip")

                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_attention_mask,
                        layer_head_mask=(
                            head_mask[idx] if head_mask is not None else None
                        ),
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        att_mode=mode,
                    )

            elif "drop" in skip_mode:

                if idx >= start_drop_layer and idx < end_layer:
                    layer_outputs = layer_outputs_prev
                    num_skips += 1
                    if verbose:
                        print(idx, "drop")
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_attention_mask,
                        layer_head_mask=(
                            head_mask[idx] if head_mask is not None else None
                        ),
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        att_mode=mode,
                    )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    att_mode=mode,
                    output_intermediate_hidden_states=output_intermediate_hidden_states,
                )

        # print(skip_mode, mode, hidden_states.shape, l, min_skip_at_len, skip_at_len, idx)

        hidden_states = layer_outputs[0]

        # if layer_outputs_prev is not None:
        #     skip_blocks_interval = 3
        # else:
        layer_outputs_prev = tuple(layer_outputs)

        if "token_index" in skip_mode:
            previous_token_index = best_token_index

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        if output_intermediate_hidden_states:
            all_hidden_states += ((hidden_states, layer_outputs[-1]),)
        elif output_hidden_states:
            all_hidden_states += (hidden_states,)

    if self.final_layer_norm is not None:
        hidden_states = self.final_layer_norm(hidden_states)

    if self.project_out is not None:
        hidden_states = self.project_out(hidden_states)

    # add hidden states from the last decoder layer

    if output_intermediate_hidden_states:
        all_hidden_states += ((hidden_states, (torch.zeros_like(hidden_states),) * 5),)
    elif output_hidden_states:
        all_hidden_states += (hidden_states,)

    if verbose:
        print(f"num_skips: {num_skips}")

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def compute_skipped_blocks(
    skip_interval=2,
    skip_mode="normal",
    start_layer=0,
    end_layer=31,
    layers_interval=[],
    exit_layer=32,
    start_drop_layer=32,
    num_layers=32,
):

    num_skips = 0

    if isinstance(skip_interval, list):
        len_skip_int = len(skip_interval)
        step = num_layers // len_skip_int

    if "drop" in skip_mode:
        return num_layers - start_drop_layer - 1

    if "exit" in skip_mode:
        return num_layers - exit_layer - 1

    for idx in range(num_layers):

        if isinstance(skip_interval, list):
            skip_interval_ = skip_interval[min(idx // step, len_skip_int - 1)]
        else:
            skip_interval_ = skip_interval

        if idx < (end_layer) and idx > (start_layer):

            if "parrallel_block" in skip_mode and (
                idx in layers_interval or (idx - 1) in layers_interval
            ):
                if (idx - 1) in layers_interval:
                    num_skips += 1
            elif idx % skip_interval_ == 0:
                num_skips += 1

    return num_skips


def forward_opt_layer(
    hidden_states: torch.Tensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    att_mode="normal",
    self=None,
    output_intermediate_hidden_states=False,
    prompt_len=10,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
            `(encoder_attention_heads,)`.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """

    residual = hidden_states
    # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
    if self.do_layer_norm_before:
        hidden_states = self.self_attn_layer_norm(hidden_states)

    if att_mode == "skip_attn":
        if past_key_value is not None:

            # after the fix
            bsz, seq_len, _ = hidden_states.shape
            key_states = (
                hidden_states.view(
                    bsz, seq_len, self.self_attn.num_heads, self.self_attn.head_dim
                )
                .transpose(1, 2)
                .contiguous()
            )
            value_states = (
                hidden_states.view(
                    bsz, seq_len, self.self_attn.num_heads, self.self_attn.head_dim
                )
                .transpose(1, 2)
                .contiguous()
            )
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

            # # before the fix
            # key_states = torch.cat([past_key_value[0], past_key_value[0][:, :, -1, :].unsqueeze(2)], dim=2)
            # value_states = torch.cat([past_key_value[1], past_key_value[1][:, :, -1, :].unsqueeze(2)], dim=2)

            present_key_value = (key_states, value_states)
        else:
            bsz, seq_len, _ = hidden_states.shape
            key_states = (
                hidden_states.view(
                    bsz, seq_len, self.self_attn.num_heads, self.self_attn.head_dim
                )
                .transpose(1, 2)
                .contiguous()
            )
            present_key_value = (key_states, key_states)

        # present_key_value = past_key_value
        hidden_states = hidden_states

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

    elif att_mode == "parrallel":

        # Self Attention
        hidden_states_, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states_ = nn.functional.dropout(
            hidden_states_, p=self.dropout, training=self.training
        )
        # hidden_states_ = residual + hidden_states_ # add residual one time

        # 350m applies layer norm AFTER attention noln
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states_.shape

        # hidden_states_ = hidden_states_.reshape(-1, hidden_states_.size(-1))

        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))

        # residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention noln
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        # hidden_states = hidden_states.unsqueeze(1)

        ## parrallel
        hidden_states = (
            hidden_states_ + hidden_states.view(hidden_states_shape) + residual
        )

        # 350m applies layer norm AFTER attention noln
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

    elif "skip_ffn" in att_mode:

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        # # 350m applies layer norm AFTER attention noln
        # if not self.do_layer_norm_before:
        #     hidden_states = self.self_attn_layer_norm(hidden_states)

        if "prompt" in att_mode and hidden_states.shape[1] > prompt_len:

            # Fully Connected
            hidden_states_shape = hidden_states.shape
            residual = hidden_states  # .reshape(-1, hidden_states.size(-1))

            hidden_states_ = hidden_states[:, prompt_len:, :]

            hidden_states__shape = hidden_states_.shape
            hidden_states_ = hidden_states_.reshape(-1, hidden_states_.size(-1))

            # hidden_states_, fc1 = self.mlp(hidden_states_)
            # hidden_states = torch.cat((hidden_states[:, :prompt_len, :], hidden_states_), dim=1)

            # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
            if self.do_layer_norm_before:
                hidden_states_ = self.final_layer_norm(hidden_states_)

            hidden_states_ = self.fc1(hidden_states_)
            hidden_states_ = self.activation_fn(hidden_states_)
            # if output_intermediate_hidden_states:
            #     intermediate_hidden_states.append(hidden_states_)

            hidden_states_ = self.fc2(hidden_states_)

            # if output_intermediate_hidden_states:
            #     intermediate_hidden_states.append(hidden_states)

            hidden_states_ = nn.functional.dropout(
                hidden_states_, p=self.dropout, training=self.training
            )

            # 350m applies layer norm AFTER attention
            if not self.do_layer_norm_before:
                hidden_states_ = self.final_layer_norm(hidden_states_)

            hidden_states_ = hidden_states_.view(hidden_states__shape)
            hidden_states = torch.cat(
                (hidden_states[:, :prompt_len, :], hidden_states_), dim=1
            )

            hidden_states = residual + hidden_states  # .view(hidden_states_shape)

        else:

            # Fully Connected
            hidden_states_shape = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
            residual = hidden_states

            hidden_states = (residual).view(hidden_states_shape)

        # # 350m applies layer norm AFTER attention noln
        # if not self.do_layer_norm_before:
        #     hidden_states = self.final_layer_norm(hidden_states)

    else:
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        if output_intermediate_hidden_states:
            tmp = hidden_states
            im, txt = tmp[:, :10, :], tmp[:, 10:, :]  # assume visual prompt len 10
            hs = torch.cat([im.mean(1, keepdim=True), txt.mean(1, keepdim=True)], dim=1)
            intermediate_hidden_states = [hs]

        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        if output_intermediate_hidden_states:
            tmp = hidden_states
            im, txt = tmp[:, :10, :], tmp[:, 10:, :]  # assume visual prompt len 10
            hs = torch.cat([im.mean(1, keepdim=True), txt.mean(1, keepdim=True)], dim=1)
            intermediate_hidden_states.append(hs)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        if output_intermediate_hidden_states:
            tmp = hidden_states.view(
                [hidden_states_shape[0], -1, hidden_states.shape[-1]]
            )
            im, txt = tmp[:, :10, :], tmp[:, 10:, :]  # assume visual prompt len 10
            hs = torch.cat([im.mean(1, keepdim=True), txt.mean(1, keepdim=True)], dim=1)
            intermediate_hidden_states.append(hs)

        hidden_states = self.fc2(hidden_states)

        if output_intermediate_hidden_states:
            tmp = hidden_states.view(hidden_states_shape)
            im, txt = tmp[:, :10, :], tmp[:, 10:, :]  # assume visual prompt len 10
            hs = torch.cat([im.mean(1, keepdim=True), txt.mean(1, keepdim=True)], dim=1)
            intermediate_hidden_states.append(hs)

        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        if output_intermediate_hidden_states:
            tmp = hidden_states
            im, txt = tmp[:, :10, :], tmp[:, 10:, :]  # assume visual prompt len 10
            hs = torch.cat([im.mean(1, keepdim=True), txt.mean(1, keepdim=True)], dim=1)
            intermediate_hidden_states.append(hs)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    if output_intermediate_hidden_states:

        outputs += (intermediate_hidden_states,)

    return outputs


def forward_opt(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    verbose=False,
    skip_interval=2,
    skip_mode="normal",
    ent_thresh=100,
    start_layer=0,
    end_layer=31,
    lm_head=None,
    layers_interval=[],
    exit_layer=32,
    start_drop_layer=32,
    output_intermediate_hidden_states=False,
    causal_prompt=False,
) -> Union[Tuple, CausalLMOutputWithPast]:

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model.decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        head_mask=head_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        verbose=verbose,
        skip_interval=skip_interval,
        skip_mode=skip_mode,
        ent_thresh=ent_thresh,
        start_layer=start_layer,
        end_layer=end_layer,
        lm_head=self.lm_head,
        layers_interval=layers_interval,
        exit_layer=exit_layer,
        start_drop_layer=start_drop_layer,
        output_intermediate_hidden_states=output_intermediate_hidden_states,
        causal_prompt=causal_prompt,
    )

    logits = self.lm_head(outputs[0]).contiguous()

    loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(logits.device)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


# from functools import partial


# for layer in model.model_text.model.decoder.layers:
#     layer.forward = partial(forward_opt_layer, self=layer) #partial(forward_opt_layer, self=layer)

# model.model_text.model.decoder.forward = partial(forward_opt_decoder, self=model.model_text.model.decoder)
