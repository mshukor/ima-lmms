import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import random
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union


def forward_llama_mlp(x, self=None):
    if self.config.pretraining_tp > 1:
        slice = self.intermediate_size // self.config.pretraining_tp
        gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
        up_proj_slices = self.up_proj.weight.split(slice, dim=0)
        down_proj_slices = self.down_proj.weight.split(slice, dim=1)

        gate_proj = torch.cat(
            [
                F.linear(x, gate_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ],
            dim=-1,
        )
        up_proj = torch.cat(
            [F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)],
            dim=-1,
        )

        intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
        down_proj = [
            F.linear(intermediate_states[i], down_proj_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        down_proj = sum(down_proj)
    else:
        fc1 = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        down_proj = self.down_proj(fc1)

    return down_proj, fc1


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


def forward_llama_layer(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    att_mode="normal",
    self=None,
    output_intermediate_hidden_states=False,
    prompt_len=10,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*):
            attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
            query_sequence_length, key_sequence_length)` if default attention is used.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """
    if "padding_mask" in kwargs:
        print(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)
    if output_intermediate_hidden_states:
        tmp = hidden_states
        im, txt = tmp[:, :10, :], tmp[:, 10:, :]  # assume visual prompt len 10
        hs = torch.cat([im.mean(1, keepdim=True), txt.mean(1, keepdim=True)], dim=1)
        intermediate_hidden_states = [hs]

    if att_mode == "parrallel":
        # Self Attention
        hidden_states_, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        # Fully Connected
        # hidden_states_ = self.post_attention_layernorm(hidden_states_) # parrallel_attn_2
        # hidden_states = self.post_attention_layernorm(hidden_states) # noln
        hidden_states, fc1 = self.mlp(hidden_states)

        hidden_states = residual + hidden_states + hidden_states_

    else:
        if att_mode == "skip_attn":

            if past_key_value is not None:

                bsz, seq_len, _ = hidden_states.shape
                key_states = hidden_states.view(
                    bsz,
                    seq_len,
                    self.self_attn.num_key_value_heads,
                    self.self_attn.head_dim,
                ).transpose(
                    1, 2
                )  # .contiguous()
                value_states = hidden_states.view(
                    bsz,
                    seq_len,
                    self.self_attn.num_key_value_heads,
                    self.self_attn.head_dim,
                ).transpose(
                    1, 2
                )  # .contiguous()

                kv_seq_len = key_states.shape[-2]
                if past_key_value is not None:
                    kv_seq_len += past_key_value[0].shape[-2]

                cos, sin = self.self_attn.rotary_emb(value_states, seq_len=kv_seq_len)

                key_states = apply_rotary_pos_emb(key_states, cos, sin, position_ids)

                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

                present_key_value = (key_states, value_states)
            else:

                bsz, seq_len, _ = hidden_states.shape
                key_states = hidden_states.view(
                    bsz,
                    seq_len,
                    self.self_attn.num_key_value_heads,
                    self.self_attn.head_dim,
                ).transpose(
                    1, 2
                )  # .contiguous()
                value_states = hidden_states.view(
                    bsz,
                    seq_len,
                    self.self_attn.num_key_value_heads,
                    self.self_attn.head_dim,
                ).transpose(
                    1, 2
                )  # .contiguous()

                kv_seq_len = key_states.shape[-2]

                cos, sin = self.self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
                key_states = apply_rotary_pos_emb(key_states, cos, sin, position_ids)

                present_key_value = (key_states, value_states)

        else:
            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

            if output_intermediate_hidden_states:
                tmp = hidden_states
                im, txt = tmp[:, :10, :], tmp[:, 10:, :]  # assume visual prompt len 10
                hs = torch.cat(
                    [im.mean(1, keepdim=True), txt.mean(1, keepdim=True)], dim=1
                )
                intermediate_hidden_states.append(hs)

            hidden_states = residual + hidden_states

            if output_intermediate_hidden_states:
                tmp = hidden_states
                im, txt = tmp[:, :10, :], tmp[:, 10:, :]  # assume visual prompt len 10
                hs = torch.cat(
                    [im.mean(1, keepdim=True), txt.mean(1, keepdim=True)], dim=1
                )
                intermediate_hidden_states.append(hs)

            # Fully Connected
            residual = hidden_states
            if "skip_ffn" not in att_mode or (
                "skip_ffn" in att_mode and "prompt" in att_mode
            ):  # noln uncomment
                hidden_states = self.post_attention_layernorm(hidden_states)

        if output_intermediate_hidden_states:
            tmp = hidden_states
            im, txt = tmp[:, :10, :], tmp[:, 10:, :]  # assume visual prompt len 10
            hs = torch.cat([im.mean(1, keepdim=True), txt.mean(1, keepdim=True)], dim=1)
            intermediate_hidden_states.append(hs)

        if "skip_ffn" not in att_mode:
            hidden_states, fc1 = self.mlp(hidden_states)
        elif "prompt" in att_mode and hidden_states.shape[1] > prompt_len:
            hidden_states_ = hidden_states[:, prompt_len:, :]
            hidden_states_, fc1 = self.mlp(hidden_states_)
            hidden_states = torch.cat(
                (hidden_states[:, :prompt_len, :], hidden_states_), dim=1
            )
        else:
            fc1 = hidden_states

        if output_intermediate_hidden_states:
            tmp = fc1
            im, txt = tmp[:, :10, :], tmp[:, 10:, :]  # assume visual prompt len 10
            hs = torch.cat([im.mean(1, keepdim=True), txt.mean(1, keepdim=True)], dim=1)
            intermediate_hidden_states.append(hs)

        if output_intermediate_hidden_states:
            tmp = hidden_states
            im, txt = tmp[:, :10, :], tmp[:, 10:, :]  # assume visual prompt len 10
            hs = torch.cat([im.mean(1, keepdim=True), txt.mean(1, keepdim=True)], dim=1)
            intermediate_hidden_states.append(hs)

        if "skip_ffn" not in att_mode or (
            "skip_ffn" in att_mode and "prompt" in att_mode
        ):  # noln uncomment
            hidden_states = residual + hidden_states

    if output_intermediate_hidden_states:
        tmp = hidden_states
        im, txt = tmp[:, :10, :], tmp[:, 10:, :]  # assume visual prompt len 10
        hs = torch.cat([im.mean(1, keepdim=True), txt.mean(1, keepdim=True)], dim=1)
        intermediate_hidden_states.append(hs)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    if output_intermediate_hidden_states:
        outputs += (intermediate_hidden_states,)

    return outputs


def forward_llama_decoder(
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
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
    output_intermediate_hidden_states=False,
    unimodal_attention_mask=False,
    stop_unimodal_attention_mask=1,
    prompt_len=10,
) -> Union[Tuple, BaseModelOutputWithPast]:
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
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if getattr(self.config, "_flash_attn_2_enabled", False):
        # 2d mask is passed through the layers
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    if unimodal_attention_mask:

        if seq_length > 1:  # training mode
            first_to_last_mask = attention_mask.clone()
            first_to_last_mask[:, :, :prompt_len, prompt_len:] = torch.finfo(
                inputs_embeds.dtype
            ).min

            last_to_first_mask = attention_mask.clone()
            last_to_first_mask[:, :, prompt_len:, :prompt_len] = torch.finfo(
                inputs_embeds.dtype
            ).min

            # Combined mask (both restrictions)
            attention_mask_uni = first_to_last_mask + last_to_first_mask
        else:
            attention_mask_uni = attention_mask.clone()
            attention_mask_uni[:, :, :, :prompt_len] = torch.finfo(
                inputs_embeds.dtype
            ).min

    # embed positions
    hidden_states = inputs_embeds

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

    layer_outputs_prev = None
    num_skips = 0

    if isinstance(skip_interval, list):
        len_skip_int = len(skip_interval)
        step = len(self.layers) // len_skip_int

    previous_token_index = None

    if output_intermediate_hidden_states:
        all_hidden_states += ((hidden_states, (torch.zeros_like(hidden_states),) * 7),)
    elif output_hidden_states:
        all_hidden_states += (hidden_states,)

    for idx, decoder_layer in enumerate(self.layers):

        if isinstance(skip_interval, list):
            skip_interval_ = skip_interval[min(idx // step, len_skip_int - 1)]
        else:
            skip_interval_ = skip_interval

        if "random" in skip_mode:
            skip_interval_ = random.choice(list(range(2, skip_interval_ + 1)))

        if verbose:
            print(idx, skip_interval_, skip_interval)

        # if output_hidden_states:
        #     all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        best_token_index = 0
        previous_token_index = 1
        ent = 100

        if idx < stop_unimodal_attention_mask and unimodal_attention_mask:
            attention_mask_ = attention_mask_uni
        else:
            attention_mask_ = attention_mask

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask_,
                position_ids,
                past_key_value,
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

            if (
                (l <= skip_at_len and l > min_skip_at_len)
                and layer_outputs_prev is not None
                and idx < (end_layer)
                and idx > (start_layer)
            ):  # len(self.layers)-1

                if idx % skip_interval_ == 0:

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
                                attention_mask=attention_mask_,
                                position_ids=position_ids,
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

                elif idx % skip_interval_ == 0:
                    layer_outputs = layer_outputs_prev
                    num_skips += 1
                    if verbose:
                        print(idx, "skip")

                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask_,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        att_mode=mode,
                    )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask_,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    output_intermediate_hidden_states=output_intermediate_hidden_states,
                    att_mode=mode,
                )

        hidden_states = layer_outputs[0]

        layer_outputs_prev = tuple(layer_outputs)

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        if output_intermediate_hidden_states:
            all_hidden_states += ((hidden_states, layer_outputs[-1]),)
        elif output_hidden_states:
            all_hidden_states += (hidden_states,)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_intermediate_hidden_states:
        all_hidden_states += ((hidden_states, (torch.zeros_like(hidden_states),) * 7),)
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


def forward_llama(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    verbose=False,
    skip_interval=40,
    skip_mode="normal",
    ent_thresh=100,
    start_layer=0,
    end_layer=31,
    lm_head=None,
    layers_interval=[],
    output_intermediate_hidden_states=False,
    unimodal_attention_mask=False,
    stop_unimodal_attention_mask=1,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

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
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
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
        output_intermediate_hidden_states=output_intermediate_hidden_states,
        unimodal_attention_mask=unimodal_attention_mask,
        stop_unimodal_attention_mask=stop_unimodal_attention_mask,
    )

    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(
            self.vocab_size // self.config.pretraining_tp, dim=0
        )
        logits = [
            F.linear(hidden_states, lm_head_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

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
