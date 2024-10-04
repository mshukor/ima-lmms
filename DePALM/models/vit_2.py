import torch
from torch import nn

import timm


class TimmViT(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained=False,
        return_hidden_state=False,
        n=6,
        img_size=224,
    ) -> None:
        super().__init__()

        self.return_hidden_state = return_hidden_state
        print(f"load {model_name} with image size {img_size}")

        self.model_name = model_name
        if "audio" in self.model_name:
            # "hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m"
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
            )
        else:
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, img_size=img_size
            )

        self.reshape = False  # useful for cnns
        self.embed_dim = self.model.embed_dim
        self.n = n

    def forward(self, x: torch.Tensor, external_features=None) -> torch.Tensor:
        all_hidden_states = () if self.return_hidden_state else None

        if "audio" in self.model_name:
            x = x.unsqueeze(1)
        # self.n = len(self.model.blocks)
        out = self.model.get_intermediate_layers(
            x,
            n=len(self.model.blocks),
            reshape=self.reshape,
            return_prefix_tokens=True,
            norm=True,
        )

        for o in out:
            tmp = torch.cat(
                (o[1], o[0]), dim=1
            )  # get_intermediate_layers return (torch.Size([1, 576, 1024]), torch.Size([1, 0, 1024]))
            all_hidden_states = all_hidden_states + (tmp,)

        if self.return_hidden_state:
            return x, all_hidden_states
        else:
            return x
