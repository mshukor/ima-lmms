import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from models.connections.keepbest import EViT, KeepBest, KeepBestClusters
from models.connections.tome import ToMe
from models.interaction import attention, tile
from torch import nn

# from models.trans import CustomTransformerEncoder


class FiLM_(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """

    def __init__(self, residual=True, use_attention=True, gate=True):
        super(FiLM_, self).__init__()

        self.residual = residual
        self.use_attention = use_attention
        if gate:
            self.gate = torch.nn.Parameter(torch.zeros(1, 1, 1))
        else:
            self.gate = None

        print(f"FiLM residual: {residual} use_attention: {use_attention},")

    def forward(self, query, vis):

        gammas = vis  # vis bs, l, dim -> bs, 1, dim

        bs_v, bs_t = gammas.shape[0], query.shape[0]
        if bs_v != bs_t:
            gammas = tile(gammas, 0, bs_t // bs_v)

        if self.use_attention:
            Q = query.unsqueeze(-2)  # b, l1, 1, d
            K = gammas.unsqueeze(1).expand(
                Q.shape[:2] + gammas.shape[1:]
            )  # b, l1, l2, d

            _, A = attention(Q, K, K, return_weights=True)  # b, l1, 1, l2

            gammas = torch.matmul(
                A, gammas.unsqueeze(1)
            ).squeeze()  # gamma bs, l2, d -> bs, 1, l2, d -> bs, l1, 1, d -> bs, l1, d
            # betas = torch.matmul(A, betas.unsqueeze(1)).squeeze()

        x = query * gammas  # + betas

        if self.residual:
            if self.gate is not None:
                x = x * self.gate.tanh() + query
            else:
                x = x + query

        filmed_txt = x

        return filmed_txt


class InterleavedTransFormer(nn.Module):
    def __init__(
        self,
        layer,
        num_trans=1,
        cat_with_text=False,
        with_film=False,
        interleaved_with_text=True,
        ca_with_text=False,
        img_with_text=False,
        config=None,
    ) -> None:
        super().__init__()
        self.num_trans = num_trans

        self.img_with_text = img_with_text

        if ca_with_text:
            self.trans = nn.ModuleList(
                [nn.TransformerDecoder(layer, num_layers=1) for i in range(num_trans)]
            )
            if self.img_with_text:
                self.img_trans = nn.ModuleList(
                    [
                        nn.TransformerDecoder(layer, num_layers=1)
                        for i in range(num_trans // 2)
                    ]
                )
        else:
            self.trans = nn.ModuleList(
                [nn.TransformerEncoder(layer, num_layers=1) for i in range(num_trans)]
            )
            if self.img_with_text:
                self.img_trans = nn.ModuleList(
                    [
                        nn.TransformerEncoder(layer, num_layers=1)
                        for i in range(num_trans // 2)
                    ]
                )

        self.cat_with_text = cat_with_text
        self.interleaved_with_text = interleaved_with_text
        self.with_film = with_film
        if self.with_film:
            self.film = FiLM_(residual=True, use_attention=True, gate=True)

        self.ca_with_text = ca_with_text

        self.token_pruning_config = (
            config.get("token_pruning_config", None) if config else None
        )
        self.t_prune = get_token_pruning(self.token_pruning_config)
        if self.token_pruning_config:
            self.prune_steps = self.token_pruning_config.get("steps", 1)
            self.token_prune_target = self.token_pruning_config.get(
                "token_prune_target", "img"
            )

        # if self.attn_qprune:
        #     KeepBest()

        self.num_sampled_img_tokens = config.get("num_sampled_img_tokens", 0)
        self.keep_img_cls = config.get("keep_img_cls", True)

        self.num_sampled_text_tokens = config.get("num_sampled_text_tokens", 0)
        self.keep_text_cls = config.get("keep_text_cls", True)

        print(
            "Load InterleavedTransFormer: ",
            "cat_with_text ",
            cat_with_text,
            "num_trans",
            num_trans,
            "with_film",
            with_film,
            "interleaved_with_text",
            interleaved_with_text,
            "ca_with_text",
            ca_with_text,
            "token_pruning_config",
            self.token_pruning_config,
            "img_with_tex",
            self.img_with_text,
            "num_sampled_img_tokens",
            self.num_sampled_img_tokens,
            "num_sampled_text_tokens",
            self.num_sampled_text_tokens,
        )

    def forward(
        self,
        x_: torch.Tensor,
    ) -> torch.Tensor:

        q, x1, x2 = x_  # q, x1, x2 = vis, txt

        if isinstance(self.num_sampled_img_tokens, list):
            L = self.num_sampled_img_tokens
            num_sampled_img_tokens = random.choice(np.arange(L[0], L[1], 0.1).tolist())
        else:
            num_sampled_img_tokens = self.num_sampled_img_tokens

        if num_sampled_img_tokens > 0 and self.training:
            num_sampled_img_tokens_ = int(num_sampled_img_tokens * x1.shape[1])

            if self.keep_img_cls:
                sampled_tokens = sample_tokens(
                    x1[:, 1:, :], num_tokens=num_sampled_img_tokens_
                )
                x1 = torch.cat([x1[:, 0, :].unsqueeze(1), sampled_tokens], dim=1)
            else:
                x1 = sample_tokens(x1, num_tokens=num_sampled_img_tokens_)

        if self.num_sampled_text_tokens > 0 and self.training:
            num_sampled_text_tokens = int(self.num_sampled_text_tokens * x2.shape[1])

            if self.keep_text_cls:
                sampled_tokens = sample_tokens(
                    x2[:, 1:, :], num_tokens=num_sampled_text_tokens
                )
                x2 = torch.cat([x2[:, 0, :].unsqueeze(1), sampled_tokens], dim=1)
            else:
                x2 = sample_tokens(x2, num_tokens=num_sampled_text_tokens)

        prune_step = 0
        if self.t_prune:
            if self.token_prune_target != "query":
                for i in range(self.prune_steps):
                    x1 = self.t_prune(x1)

        if self.img_with_text:
            for i in range(len(self.img_trans)):
                if self.ca_with_text:
                    x1 = self.img_trans[i](x1, x2)
                else:
                    out = self.img_trans[i](torch.cat((x2, x1), dim=1))
                    x2, x1 = out[:, : -x1.size(1)], out[:, -x1.size(1) :]

        x = [x1, x2]

        for i in range(self.num_trans):
            if self.with_film:
                q = self.film(q, x2)

            if self.cat_with_text:
                x_ = torch.cat((x[0], x[1]), dim=1)
            elif self.interleaved_with_text:
                idx = i % 2
                x_ = x[idx]
            else:
                x_ = x[0]

            if self.ca_with_text:
                q = self.trans[i](q, x_)
            else:
                out = self.trans[i](torch.cat((x_, q), dim=1))
                x_, q = out[:, : -q.size(1)], out[:, -q.size(1) :]

            if self.t_prune:
                if self.token_prune_target == "query":
                    if prune_step < self.prune_steps:
                        q = self.t_prune(q)
                        prune_step += 1

        # if self.attn_prune:
        #     with torch.no_grad():
        #         _, att_weights = self.trans[-1].layers[-1].self_attn(q, mem, mem, need_weights=True, average_attn_weights=True) # (bs, lq, lkv) multihead_attn for decoder
        #         q_score = att_weights.mean(-1)
        #         rate = torch.sum(q_score>0.3).item()
        #         self.attn_prune(x, scores=q_score, rate=rate) # problem how to pad queies with variyng size in the batch

        return q


from typing import Optional

from torch import Tensor


def _generate_square_subsequent_mask(
    sz: int,
    device: torch.device = torch.device(
        torch._C._get_default_device()
    ),  # torch.device('cpu'),
    dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    r"""Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )


def _get_seq_len(src: Tensor, batch_first: bool) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _detect_is_causal_mask(
    mask: Optional[Tensor],
    is_causal: Optional[bool] = None,
    size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = is_causal is True

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype
        )

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


def forward_TransformerEncoder(
    src: Tensor,
    mask: Optional[Tensor] = None,
    src_key_padding_mask: Optional[Tensor] = None,
    self=None,
) -> Tensor:
    r"""Pass the input through the encoder layers in turn.

    Args:
        src: the sequence to the encoder (required).
        mask: the mask for the src sequence (optional).
        src_key_padding_mask: the mask for the src keys per batch (optional).

    Shape:
        see the docs in Transformer class.
    """
    if src_key_padding_mask is not None:
        _skpm_dtype = src_key_padding_mask.dtype
        if _skpm_dtype != torch.bool and not torch.is_floating_point(
            src_key_padding_mask
        ):
            raise AssertionError(
                "only bool and floating types of key_padding_mask are supported"
            )
    output = src
    convert_to_nested = False
    first_layer = self.layers[0]
    src_key_padding_mask_for_layers = src_key_padding_mask
    why_not_sparsity_fast_path = ""
    str_first_layer = "self.layers[0]"
    if not isinstance(first_layer, torch.nn.TransformerEncoderLayer):
        why_not_sparsity_fast_path = (
            f"{str_first_layer} was not TransformerEncoderLayer"
        )
    elif first_layer.norm_first:
        why_not_sparsity_fast_path = f"{str_first_layer}.norm_first was True"
    elif first_layer.training:
        why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
    elif not first_layer.self_attn.batch_first:
        why_not_sparsity_fast_path = (
            f" {str_first_layer}.self_attn.batch_first was not True"
        )
    elif not first_layer.self_attn._qkv_same_embed_dim:
        why_not_sparsity_fast_path = (
            f"{str_first_layer}.self_attn._qkv_same_embed_dim was not True"
        )
    elif not first_layer.activation_relu_or_gelu:
        why_not_sparsity_fast_path = (
            f" {str_first_layer}.activation_relu_or_gelu was not True"
        )
    elif not (first_layer.norm1.eps == first_layer.norm2.eps):
        why_not_sparsity_fast_path = (
            f"{str_first_layer}.norm1.eps was not equal to {str_first_layer}.norm2.eps"
        )
    elif not src.dim() == 3:
        why_not_sparsity_fast_path = (
            f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        )
    elif not self.enable_nested_tensor:
        why_not_sparsity_fast_path = "enable_nested_tensor was not True"
    elif src_key_padding_mask is None:
        why_not_sparsity_fast_path = "src_key_padding_mask was None"
    elif (
        (not hasattr(self, "mask_check")) or self.mask_check
    ) and not torch._nested_tensor_from_mask_left_aligned(
        src, src_key_padding_mask.logical_not()
    ):
        why_not_sparsity_fast_path = (
            "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        )
    elif output.is_nested:
        why_not_sparsity_fast_path = "NestedTensor input is not supported"
    elif mask is not None:
        why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
    elif first_layer.self_attn.num_heads % 2 == 1:
        why_not_sparsity_fast_path = "num_head is odd"
    elif torch.is_autocast_enabled():
        why_not_sparsity_fast_path = "autocast is enabled"

    if not why_not_sparsity_fast_path:
        tensor_args = (
            src,
            first_layer.self_attn.in_proj_weight,
            first_layer.self_attn.in_proj_bias,
            first_layer.self_attn.out_proj.weight,
            first_layer.self_attn.out_proj.bias,
            first_layer.norm1.weight,
            first_layer.norm1.bias,
            first_layer.norm2.weight,
            first_layer.norm2.bias,
            first_layer.linear1.weight,
            first_layer.linear1.bias,
            first_layer.linear2.weight,
            first_layer.linear2.bias,
        )

        if torch.overrides.has_torch_function(tensor_args):
            why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
        elif not (src.is_cuda or "cpu" in str(src.device)):
            why_not_sparsity_fast_path = "src is neither CUDA nor CPU"
        elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
            why_not_sparsity_fast_path = (
                "grad is enabled and at least one of query or the "
                "input/output projection weights or biases requires_grad"
            )

        if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
            convert_to_nested = True
            output = torch._nested_tensor_from_mask(
                output, src_key_padding_mask.logical_not(), mask_check=False
            )
            src_key_padding_mask_for_layers = None

    all_outputs = []

    if self.norm is not None:
        all_outputs.append(self.norm(output))
    else:
        all_outputs.append(output)

    for mod in self.layers:
        output = mod(
            output, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers
        )
        if self.norm is not None:
            all_outputs.append(self.norm(output))
        else:
            all_outputs.append(output)

    if convert_to_nested:
        output = output.to_padded_tensor(0.0)

    if self.norm is not None:
        output = self.norm(output)

    return output, all_outputs


class TransFormer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config=None,
    ) -> None:
        super().__init__()

        hidden_dim = config.get("hidden_dim", 256)
        num_heads = config.get("num_heads", 8)
        num_layers = config.get("num_layers", 4)
        output_length = config.get("output_length", 32)
        activation = config.get("activation", "relu")

        self.q_downsample = config.get("q_downsample", False)

        self.qs_former = config.get("qs_former", True)
        self.with_film = config.get("with_film", False)
        self.block = config.get("block", "sa")
        self.use_upsample = config.get("use_upsample", True)

        self.with_cat_film = config.get("with_cat_film", False)

        self.interleaved_qs_former = config.get("interleaved_qs_former", False)
        self.cat_with_text = config.get("cat_with_text", False)
        self.number_connector_heads = config.get("number_connector_heads", 1)
        self.qs_former_with_film = config.get("qs_former_with_film", False)
        self.interleaved_with_text = config.get("interleaved_with_text", True)
        self.ca_with_text = config.get("ca_with_text", False)
        self.img_with_text = config.get("img_with_text", False)
        self.config_qs_former = config  # config.get('config_qs_former', {})

        self.output_hidden_states = config.get("output_hidden_states", False)

        print(
            "qs_former: ",
            self.qs_former,
            "number_connector_heads",
            self.number_connector_heads,
        )
        if activation == "relu":
            activation = F.relu
        elif activation == "gelu":
            activation = F.gelu
        else:
            raise NotImplemented

        self.down = nn.Linear(
            input_dim,
            hidden_dim,
        )
        if self.interleaved_qs_former:
            self.down1 = nn.Linear(
                output_dim,
                hidden_dim,
            )

        if self.use_upsample:
            self.up = nn.Linear(
                hidden_dim,
                input_dim,
            )
            trans_out_dim = input_dim
        else:
            trans_out_dim = hidden_dim

        if self.q_downsample:
            self.q_down = nn.Linear(
                output_dim,
                hidden_dim,
            )

        if self.block == "ca":
            print("Cross-attention connector")
            encoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=2 * hidden_dim,
                dropout=0.0,
                activation=activation,
                batch_first=True,
                norm_first=True,
            )
            self.trans = nn.TransformerDecoder(encoder_layer, num_layers=num_layers)

        # elif self.block == 'trans_film':
        #     encoder_layer = nn.TransformerEncoderLayer(
        #         d_model=hidden_dim,
        #         nhead=num_heads,
        #         dim_feedforward=2 * hidden_dim,
        #         dropout=0.,
        #         activation=activation,
        #         batch_first=True,
        #         norm_first=True
        #     )
        #     self.trans = CustomTransformerEncoder(encoder_layer, num_layers=num_layers)
        elif self.interleaved_qs_former:

            if self.ca_with_text:
                encoder_layer = nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=2 * hidden_dim,
                    dropout=0.0,
                    activation=activation,
                    batch_first=True,
                    norm_first=True,
                )
            else:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=2 * hidden_dim,
                    dropout=0.0,
                    activation=activation,
                    batch_first=True,
                    norm_first=True,
                )

            self.trans = InterleavedTransFormer(
                encoder_layer,
                num_trans=num_layers,
                cat_with_text=self.cat_with_text,
                with_film=self.qs_former_with_film,
                config=self.config_qs_former,
                interleaved_with_text=self.interleaved_with_text,
                ca_with_text=self.ca_with_text,
                img_with_text=self.img_with_text,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=2 * hidden_dim,
                dropout=0.0,
                activation=activation,
                batch_first=True,
                norm_first=True,
            )
            self.trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            if self.output_hidden_states:
                print("Load custon forward")
                self.trans.forward = partial(
                    forward_TransformerEncoder, self=self.trans
                )  # partial(forward_TransformerEncoder, self=self.trans)

        if self.qs_former or self.block == "ca":
            if self.q_downsample:
                self.const = nn.Parameter(torch.randn(output_length, output_dim))
            else:
                self.const = nn.Parameter(torch.randn(output_length, hidden_dim))

        # if self.number_connector_heads == 1:
        #     if self.with_film:
        #         self.gammas_func, self.betas_func, self.gammas_gate, self.betas_gate = self.get_connector_head(trans_out_dim, output_dim, config)
        #     else:
        #         self.proj = self.get_connector_head(trans_out_dim, output_dim, config)
        # else:
        if self.with_film:
            (
                self.gammas_func,
                self.betas_func,
                self.gammas_gate,
                self.betas_gate,
            ) = self.get_connector_head(
                trans_out_dim, int(output_dim * self.number_connector_heads), config
            )
        elif self.with_cat_film:
            config["proj_out_dim"] = output_dim
            (
                self.proj,
                self.gammas_func,
                self.betas_func,
                self.gammas_gate,
                self.betas_gate,
            ) = self.get_connector_head(
                trans_out_dim, int(output_dim * self.number_connector_heads), config
            )
        else:
            self.proj = self.get_connector_head(
                trans_out_dim, int(output_dim * self.number_connector_heads), config
            )

    def get_connector_head(self, input_dim, output_dim, config):

        if self.with_film or self.with_cat_film:

            gamma_beta_type = config.get("gamma_beta_type", "linear")
            print("build film connector with: ", config)
            if gamma_beta_type == "linear":
                gammas_func = nn.Linear(input_dim, output_dim)
                betas_func = nn.Linear(input_dim, output_dim)
            elif gamma_beta_type == "lora":
                rank = config.get("rank", 2)
                min_dim = min(input_dim, output_dim)
                gammas_func = nn.Sequential(
                    nn.Linear(input_dim, min_dim // rank),
                    nn.Linear(min_dim // rank, output_dim),
                )
                betas_func = nn.Sequential(
                    nn.Linear(input_dim, min_dim // rank),
                    nn.Linear(min_dim // rank, output_dim),
                )
            elif gamma_beta_type == "lora_relu":
                rank = config.get("rank", 2)
                min_dim = min(input_dim, output_dim)
                gammas_func = nn.Sequential(
                    nn.Linear(input_dim, min_dim // rank),
                    nn.ReLU(),
                    nn.Linear(min_dim // rank, output_dim),
                )
                betas_func = nn.Sequential(
                    nn.Linear(input_dim, min_dim // rank),
                    nn.ReLU(),
                    nn.Linear(min_dim // rank, output_dim),
                )
            else:
                raise NotImplemented
            if self.number_connector_heads > 1:
                gammas_gate = torch.nn.Parameter(
                    torch.zeros(1, 1, self.number_connector_heads, 1)
                )
                betas_gate = torch.nn.Parameter(
                    torch.zeros(1, 1, self.number_connector_heads, 1)
                )
            else:
                gammas_gate = torch.nn.Parameter(torch.zeros(1, 1, 1))
                betas_gate = torch.nn.Parameter(torch.zeros(1, 1, 1))

            if self.with_cat_film:
                proj = nn.Linear(input_dim, config["proj_out_dim"])
                return proj, gammas_func, betas_func, gammas_gate, betas_gate
            return gammas_func, betas_func, gammas_gate, betas_gate
        else:
            proj = nn.Linear(input_dim, output_dim)
            return proj

    def forward(self, x_) -> torch.Tensor:

        # if self.cat_text:
        #     x_txt = x_[1]
        #     x = x_[0]

        if isinstance(x_, list) and not self.interleaved_qs_former:
            query = x_[1]
            x = x_[0]
        else:
            query = None
            x = x_

        if self.interleaved_qs_former:
            x0 = self.down(x[0])
            x1 = self.down1(x[1])
            query = self.const.unsqueeze(0).expand(x0.size(0), -1, -1)
            x = self.trans([query, x0, x1])  # q, vis, txt
            if self.output_hidden_states:
                x, output_hidden_states = x, [
                    x,
                ]  # not implemented
        else:

            x = self.down(x)

            main_query = self.const.unsqueeze(0).expand(x.size(0), -1, -1)
            query = query if query is not None else main_query

            if self.q_downsample:
                query = self.q_down(query)

            if self.qs_former:
                x = torch.cat((x, query), dim=1)

            if self.block == "ca":
                x = self.trans(query, x)
                if self.output_hidden_states:
                    x, output_hidden_states = x, [
                        x,
                    ]  # not implemented
            else:
                x = self.trans(x)

                if self.output_hidden_states:
                    x, output_hidden_states = x

            if self.qs_former:
                x = x[:, -query.size(1) :]

        if self.use_upsample:
            x = self.up(x)
        # print("connector 2", x.shape, query.shape)
        if self.number_connector_heads == 1:
            if self.with_film:
                gammas = (
                    self.gammas_func(x) * self.gammas_gate.tanh()
                )  # .mean(1, keepdim=True)
                betas = (
                    self.betas_func(x) * self.betas_gate.tanh()
                )  # .mean(1, keepdim=True)
                # tokens (bs, l, dim) film is on dim, here we don't have spatial dim, no need fo expand
                return [x, gammas, betas]
            else:
                if self.output_hidden_states:
                    return self.proj(x), output_hidden_states
                else:
                    return self.proj(x)
        else:
            if self.with_film or self.with_cat_film:
                gammas = self.gammas_func(x)  # .mean(1, keepdim=True) (bs, t, d*l)
                betas = self.betas_func(x)  # .mean(1, keepdim=True)
                # tokens (bs, l, dim) film is on dim, here we don't have spatial dim, no need fo expand
                b, t, d = gammas.shape
                gammas = (
                    gammas.reshape(b, t, self.number_connector_heads, -1)
                    * self.gammas_gate.tanh()
                )
                betas = (
                    betas.reshape(b, t, self.number_connector_heads, -1)
                    * self.betas_gate.tanh()
                )
                if self.with_cat_film:
                    x = self.proj(x)
                output = [
                    [x, gammas[:, :, j, :], betas[:, :, j, :]]
                    for j in range(self.number_connector_heads)
                ]
                return output
            else:
                b, t, d = x.shape
                proj = self.proj(x)
                proj = proj.reshape(b, t, self.number_connector_heads, -1)
                output = [
                    [proj[:, :, j, :]] for j in range(self.number_connector_heads)
                ]
                return output


class CNN(nn.Module):
    def __init__(self, inplanes, planes, config=None):
        super().__init__()

        stride = config.get("stride", 1)
        self.with_cls = config.get("with_cls", 1)
        self.with_film = config.get("with_film", False)

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, inplanes // 2, 3, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes // 2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(inplanes // 2, inplanes, 3, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        if self.with_film:

            gamma_beta_type = config.get("gamma_beta_type", "linear")
            print("build film connector with: ", config)
            if gamma_beta_type == "linear":
                self.gammas_func = nn.Linear(inplanes, planes)
                self.betas_func = nn.Linear(inplanes, planes)
            elif gamma_beta_type == "lora":
                rank = config.get("rank", 2)
                min_dim = min(inplanes, planes)
                self.gammas_func = nn.Sequential(
                    nn.Linear(inplanes, min_dim // rank),
                    nn.Linear(min_dim // rank, planes),
                )
                self.betas_func = nn.Sequential(
                    nn.Linear(inplanes, min_dim // rank),
                    nn.Linear(min_dim // rank, planes),
                )
            elif gamma_beta_type == "lora_relu":
                rank = config.get("rank", 2)
                min_dim = min(inplanes, planes)
                self.gammas_func = nn.Sequential(
                    nn.Linear(inplanes, min_dim // rank),
                    nn.ReLU(),
                    nn.Linear(min_dim // rank, planes),
                )
                self.betas_func = nn.Sequential(
                    nn.Linear(inplanes, min_dim // rank),
                    nn.ReLU(),
                    nn.Linear(min_dim // rank, planes),
                )
            else:
                raise NotImplemented

            self.gammas_gate = torch.nn.Parameter(torch.zeros(1, 1, 1))
            self.betas_gate = torch.nn.Parameter(torch.zeros(1, 1, 1))
        else:
            self.proj = nn.Linear(inplanes, planes)

    def forward(self, x: torch.Tensor):

        if self.with_cls:
            cls_ = x[:, :1, :]
            x_ = x[:, 1:, :]

        b, l, d = x_.shape

        x_ = x_.view(b, int(math.sqrt(l)), int(math.sqrt(l)), d)
        x_ = x_.transpose(-1, 1)
        out = self.relu1(self.bn1(self.conv1(x_)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)

        out = out.view(b, -1, out.shape[1])
        if self.with_cls:
            out = torch.cat((cls_, out), dim=1)
        if self.with_film:
            gammas = (
                self.gammas_func(out) * self.gammas_gate.tanh()
            )  # .mean(1, keepdim=True)
            betas = (
                self.betas_func(out) * self.betas_gate.tanh()
            )  # .mean(1, keepdim=True)
            # tokens (bs, l, dim) film is on dim, here we don't have spatial dim, no need fo expand
            return [out, gammas, betas]
        else:
            return self.proj(out)


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """

    def __init__(self, input_dim, output_dim, config={}):
        super(FiLM, self).__init__()

        gamma_beta_type = config.get("gamma_beta_type", "linear")
        print("build film connector with: ", config)
        if gamma_beta_type == "linear":
            self.gammas_func = nn.Linear(input_dim, output_dim)
            self.betas_func = nn.Linear(input_dim, output_dim)
        elif gamma_beta_type == "lora":
            rank = config.get("rank", 2)
            min_dim = min(input_dim, output_dim)
            self.gammas_func = nn.Sequential(
                nn.Linear(input_dim, min_dim // rank),
                nn.Linear(min_dim // rank, output_dim),
            )
            self.betas_func = nn.Sequential(
                nn.Linear(input_dim, min_dim // rank),
                nn.Linear(min_dim // rank, output_dim),
            )
        else:
            raise NotImplemented

        self.gammas_gate = torch.nn.Parameter(torch.zeros(1, 1, 1))
        self.betas_gate = torch.nn.Parameter(torch.zeros(1, 1, 1))

        self.token_pruning_config = (
            config.get("token_pruning_config", None) if config else None
        )
        self.t_prune = get_token_pruning(self.token_pruning_config)
        if self.token_pruning_config:
            self.prune_steps = self.token_pruning_config.get("steps", 1)

    def forward(self, x):
        if self.t_prune:
            for i in range(self.prune_steps):
                x = self.t_prune(x)
        gammas = self.gammas_func(x) * self.gammas_gate.tanh()  # .mean(1, keepdim=True)
        betas = self.betas_func(x) * self.betas_gate.tanh()  # .mean(1, keepdim=True)
        # tokens (bs, l, dim) film is on dim, here we don't have spatial dim, no need fo expand
        return [x, gammas, betas]
        # return x


class GatedLinear(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """

    def __init__(self, input_dim, output_dim, token_pruning_config=None):
        super(GatedLinear, self).__init__()
        self.func = nn.Linear(input_dim, output_dim)
        self.gate = torch.nn.Parameter(torch.zeros(1, 1, 1))

        self.token_pruning_config = token_pruning_config

        self.t_prune = get_token_pruning(self.token_pruning_config)
        if self.token_pruning_config:
            self.prune_steps = self.token_pruning_config.get("steps", 1)

    def forward(self, x):

        if self.t_prune:
            for i in range(self.prune_steps):
                x = self.t_prune(x)

        x = self.func(x) * self.gate.tanh()
        return x
        # return x


def sample_tokens(x, num_tokens=20):

    original_num_tokens = x.shape[1]

    patch_orders = [
        random.sample(range(original_num_tokens), k=num_tokens)
        for _ in range(x.size(0))
    ]
    patch_orders = torch.LongTensor(patch_orders).to(x.device)
    x = x.gather(1, patch_orders.unsqueeze(2).expand(-1, -1, x.size(2)))

    return x


def get_token_pruning(token_pruning_config):

    if token_pruning_config is not None:

        token_pruning_method = token_pruning_config.get("pruning_method", "tome")
        with_cls_token = token_pruning_config.get("with_cls_token", True)

        rate = token_pruning_config.get("rate", 0.5)
        if token_pruning_method == "tome":
            print("Initialize ToMe token pruning ...")
            steps = token_pruning_config.get("steps", 2)
            t_prune = ToMe(r=rate, with_cls_token=with_cls_token, steps=steps)
            return t_prune
        elif token_pruning_method == "keepbest":
            print("Initialize KeepBest token pruning ...")
            t_prune = KeepBest(r=rate)
            return t_prune
        elif token_pruning_method == "keepbestclusters":
            print("Initialize KeepBestClusters token pruning ...")
            t_prune = KeepBestClusters(clusters_size=rate)
            return t_prune

        else:
            raise NotImplemented
    else:
        return None


class Ident(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """

    def __init__(
        self,
    ):
        super(Ident, self).__init__()

    def forward(self, x):
        return x


def connector(connector_type="linear", **kwargs):
    print("Build connector:", connector_type)

    connector_config = kwargs.get("connector_config", {})
    if connector_config:
        token_pruning_config = connector_config.get("token_pruning_config", None)
    else:
        token_pruning_config = None

    if connector_type == "linear":
        return nn.ModuleList(
            [
                nn.Linear(kwargs["input_dim"], kwargs["output_dim"])
                for i in range(kwargs["num_layers"])
            ]
        )
    elif connector_type == "gated_linear":
        return nn.ModuleList(
            [
                GatedLinear(
                    kwargs["input_dim"],
                    kwargs["output_dim"],
                    token_pruning_config=token_pruning_config,
                )
                for i in range(kwargs["num_layers"])
            ]
        )
    elif connector_type == "film":
        return nn.ModuleList(
            [
                FiLM(kwargs["input_dim"], kwargs["output_dim"], config=connector_config)
                for i in range(kwargs["num_layers"])
            ]
        )

    elif connector_type == "cnn":
        return nn.ModuleList(
            [
                CNN(kwargs["input_dim"], kwargs["output_dim"], config=connector_config)
                for i in range(kwargs["num_layers"])
            ]
        )

    elif connector_type == "trans" or connector_type == "qsformer":
        return nn.ModuleList(
            [
                TransFormer(
                    kwargs["input_dim"], kwargs["output_dim"], config=connector_config
                )
                for i in range(kwargs["num_layers"])
            ]
        )

    elif connector_type == "evit":
        return nn.ModuleList(
            [
                EViT(kwargs["input_dim"], kwargs["output_dim"], config=connector_config)
                for i in range(kwargs["num_layers"])
            ]
        )
    elif connector_type == "identity":
        return nn.ModuleList([Ident() for i in range(kwargs["num_layers"])])

    else:
        raise NotImplemented
