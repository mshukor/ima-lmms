import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    )
    return torch.index_select(x, dim, order_index.to(x.device))


class CAT(nn.Module):
    """
    eP-ALM concat interaction
    """

    def __init__(self, input_dim, output_dim, **kwargs):
        super(CAT, self).__init__()

    def forward(self, txt, vis, kwargs_, **kwargs):

        tgt_len, src_len = kwargs_["tgt_len"], kwargs_["src_len"]
        token_added = kwargs_["token_added"]
        attention_mask = kwargs_["attention_mask"]

        bs, l, dim = vis.size()

        bs_v, bs_t = vis.shape[0], txt.shape[0]
        if bs_v != bs_t:
            vis = tile(vis, 0, bs_t // bs_v)

        if token_added and (tgt_len == src_len):
            token_added = True
            txt[:, :l, :] = vis[:, :l, :]
        else:
            if tgt_len == src_len:
                txt = torch.cat((vis, txt), dim=1)  # (bs, l, dim)
                attention_mask = F.pad(
                    attention_mask, (l, 0, l, 0, 0, 0, 0, 0), "constant", 0
                )
            else:
                raise
            token_added = True
        return [txt, {"attention_mask": attention_mask, "token_added": token_added}]


def interaction(config=None, **kwargs):

    interaction_type = config.get("interaction_type", "cat")
    target = config.get("target", "prompt")
    residual = config.get("residual", False)
    use_attention = config.get("use_attention_film", False)
    interaction_config = config.get("interaction_config", None)

    print(f"Building {kwargs['num_layers']} interaction layers ...")

    if isinstance(interaction_type, list):
        layers = []
        print("Build interaction: ", len(interaction_type), interaction_type)

        for i in range(kwargs["num_layers"]):
            idx = min(i, len(interaction_type) - 1)
            if interaction_type[idx] == "cat":
                inter = CAT
            else:
                raise NotImplemented
            target_ = target[idx]
            layers.append(
                inter(
                    kwargs["input_dim"],
                    kwargs["output_dim"],
                    residual=residual,
                    use_attention=use_attention,
                    target=target_,
                    config=interaction_config,
                )
            )
        interaction_type = nn.ModuleList(layers)
        return interaction_type
    else:
        print("Build interaction module:", interaction_type)
        if interaction_type == "cat":
            return nn.ModuleList(
                [
                    CAT(kwargs["input_dim"], kwargs["output_dim"])
                    for i in range(kwargs["num_layers"])
                ]
            )
        else:
            raise NotImplemented
