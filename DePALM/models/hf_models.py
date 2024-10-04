from typing import Optional

import torch
from torch import nn
from transformers import VideoMAEModel, XCLIPModel


class HFmodelwrapper(nn.Module):
    def __init__(
        self,
        model_name="xclip",
        num_frames=8,
    ):
        super().__init__()
        self.model_name = model_name
        if "xclip" in model_name:
            encoder = XCLIPModel.from_pretrained(
                "microsoft/xclip-large-patch14", local_files_only=True
            )  # 8 frames 224 / 32 frames 224 microsoft/xclip-base-patch16-zero-shot
            self.encoder_config = encoder.config
            self.encoder_vision_model = encoder.vision_model
            self.encoder_visual_projection = encoder.visual_projection
            self.encoder_mit = encoder.mit

            self.vis_dim = encoder.mit.encoder.config.hidden_size

        elif "videomae" in model_name:
            self.encoder = VideoMAEModel.from_pretrained(
                "MCG-NJU/videomae-large", num_frames=num_frames, local_files_only=True
            )
            self.vis_dim = self.encoder.config.hidden_size
            print(self.encoder.config)

        else:
            raise NotImplemented

        print(f"vis_dim {self.vis_dim}")

    def get_xclip_video_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        # Use X_CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.encoder_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.encoder_config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.encoder_config.use_return_dict
        )

        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, num_channels, height, width)

        vision_outputs = self.encoder_vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        video_embeds = vision_outputs[1]
        video_embeds = self.encoder_visual_projection(video_embeds)

        cls_features = video_embeds.view(batch_size, num_frames, -1)

        mit_outputs = self.encoder_mit(
            cls_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        video_embeds = mit_outputs[1]
        all_hidden_states = mit_outputs[2]

        return video_embeds, all_hidden_states

    def forward(self, x, **kwargs):

        if "xclip" in self.model_name:
            x, features = self.get_xclip_video_features(
                x, output_hidden_states=True, **kwargs
            )
        elif "clap" in self.model_name:
            if x.ndim < 4:
                x = x.unsqueeze(1)  # (bs, 1, l, dim)
            x = self.encoder(x, output_hidden_states=True, **kwargs)
            features = x.hidden_states
            features = [
                f.flatten(-2, -1).transpose(1, 2) for f in features
            ]  # torch.Size([1, 768, 8, 8]) ->
        else:
            x = self.encoder(x, output_hidden_states=True, **kwargs)
            features = x.hidden_states

        return x, features
