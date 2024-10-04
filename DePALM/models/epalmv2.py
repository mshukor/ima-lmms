import os
import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from models.connector import connector
from models.hf_models import HFmodelwrapper
from models.interaction import interaction
from models.vit_2 import TimmViT
from torch import nn
from transformers import LlamaForCausalLM, OPTForCausalLM


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    )
    return torch.index_select(x, dim, order_index.to(x.device))


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


## modified from https://github.com/ylsung/VL_adapter/blob/main/VL-T5/src/prompt/prompt_modeling.py


class InputPrompts(nn.Module):
    def __init__(
        self,
        prompt_len=10,
        prompt_dim=1024,
        mid_dim=512,
        mlp=True,
        deep=False,
        nb_prompts=12,
        external_input=False,
    ):
        super().__init__()

        self.prompt_len = prompt_len
        self.prompt_dim = prompt_dim
        self.mid_dim = mid_dim

        self.deep = deep
        self.nb_prompts = nb_prompts

        print(f"prompt {prompt_len}")
        self.external_input = external_input
        if self.deep:
            print("Init deep prompts", nb_prompts)
            p_len = prompt_len * nb_prompts
        else:
            p_len = prompt_len

        self.prefix_tokens = torch.arange(p_len).long()
        if mlp:
            self.prefix_embedding = nn.Sequential(
                nn.Embedding(p_len, self.prompt_dim),
                nn.Linear(self.prompt_dim, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.prompt_dim),
            )
        else:
            if self.external_input:
                self.prefix_embedding = nn.Sequential(
                    nn.Linear(prompt_dim, self.prompt_dim),
                )
            else:
                self.prefix_embedding = nn.Sequential(
                    nn.Embedding(p_len, self.prompt_dim),
                )

    def get_prompt(self, bsz, device, external_input=None):
        input_tokens = (
            self.prefix_tokens.unsqueeze(0).expand(bsz, -1).to(device)
        )  # (B, L)
        if self.external_input:
            input_tokens = (
                external_input.mean(1)
                .unsqueeze(1)
                .expand(-1, self.prompt_len, -1)
                .to(device)
            )

        prefix_prompt = self.prefix_embedding(input_tokens)  # (B, L, pdim)

        if self.deep:

            prefix_prompt = prefix_prompt.view(
                bsz, self.nb_prompts, self.prompt_len, self.prompt_dim
            )
            prompts = [prefix_prompt[:, i, :, :] for i in range(self.nb_prompts)]
            return prompts

        return prefix_prompt


class ePALMv2(nn.Module):
    def __init__(
        self,
        opt_model_name="facebook/opt-350m",
        vision_model_name="vit_base_patch16_224",
        start_layer_idx=11,
        end_layer_idx=23,
        return_hidden_state_vision=True,
        config=None,
    ):
        super().__init__()
        print("Loading ePALMv2 ...")

        self.select_higher_step = config.get("select_higher_step", False)

        self.interaction_step = config.get("interaction_step", 1)

        self.interaction_type = config.get("interaction_type", "cat")

        self.inject_outside = config.get("inject_outside", False)

        self.noise_modality = config.get("noise_modality", False)

        print("Loading: ", opt_model_name)
        self.opt_model_name = opt_model_name
        self.no_attention_mask = False

        if "opt" in opt_model_name:
            print(opt_model_name)
            self.model_text = OPTForCausalLM.from_pretrained(
                opt_model_name, torch_dtype=torch.float16, local_files_only=True
            )
        elif any([k in opt_model_name for k in ["llama", "llava", "vicuna"]]):
            print(opt_model_name)
            self.model_text = LlamaForCausalLM.from_pretrained(
                opt_model_name, torch_dtype=torch.float16, local_files_only=True
            )
        else:
            raise NotImplemented

        print("Finish reading model")
        self.transformer = self.get_llm_layers()

        print("Config")
        config_opt = self.model_text.config
        print(self.model_text.config)
        # vision
        print("Loading: ", vision_model_name)

        image_size = config.get("image_res", 224)
        num_frames = config.get("num_frames", 4)
        pretrained_model = config.get("pretrained_model", None)

        space_only_for_images = config.get("space_only_for_images", None)
        if "timesformer" in vision_model_name:
            from models.timesformer import TimeSformer

            print("Load:", pretrained_model)
            self.model_vision = TimeSformer(
                img_size=image_size,
                num_frames=num_frames,
                attention_type="divided_space_time",
                pretrained_model=pretrained_model,
                return_hidden_state=return_hidden_state_vision,
                space_only_for_images=space_only_for_images,
            )
            vis_dim = self.model_vision.embed_dim

        elif any([k in vision_model_name for k in ["xclip", "videomae"]]):
            self.model_vision = HFmodelwrapper(vision_model_name)
            vis_dim = self.model_vision.vis_dim

        elif "ast" in vision_model_name:
            from models.ast import ASTModel

            print("Load:", pretrained_model)
            self.model_vision = ASTModel(
                audioset_pretrain=True,
                verbose=True,
                pretrained_model=pretrained_model,
                return_hidden_state=return_hidden_state_vision,
            )
            vis_dim = self.model_vision.original_embedding_dim

        elif "clip" in vision_model_name:
            from models.clip import load as load_clip

            device = "cuda" if torch.cuda.is_available() else "cpu"

            if "large" in vision_model_name:
                clip_name = "ViT-L/14"
            elif "RN50x4" in vision_model_name:
                clip_name = "RN50x4"
            elif "RN50x16" in vision_model_name:
                clip_name = "RN50x16"
            else:
                clip_name = "ViT-B/16"
            print("Load ", clip_name)
            model, _ = load_clip(
                clip_name,
                device,
                return_hidden_state=True,
                download_root=os.environ["XDG_CACHE_HOME"],
            )
            self.model_vision = model.visual
            vis_dim = self.model_vision.embed_dim
            if hasattr(self.model_vision, "embed_dims"):
                injected_hidden_states = config.get("injected_hidden_states", 1)
                vis_dim = self.model_vision.embed_dims[-injected_hidden_states:]

        else:
            if pretrained_model is not None:
                pretrained = False
            else:
                pretrained = True

            self.model_vision = TimmViT(
                vision_model_name,
                pretrained=pretrained,
                return_hidden_state=return_hidden_state_vision,
                img_size=image_size,
            )

            if pretrained_model:
                self.model_vision.load_pretrained(pretrained_model)

            vis_dim = self.model_vision.embed_dim

        # connector
        connector_type = config.get("connector_type", "linear")
        self.connector_type = connector_type

        injected_hidden_states = config.get("injected_hidden_states", 1)
        self.injected_hidden_states = injected_hidden_states

        if "350" in opt_model_name:
            text_dim = self.model_text.config.word_embed_proj_dim
        else:
            text_dim = self.model_text.config.hidden_size

        connector_config = config.get("connector_config", {})
        self.shared_connector = config.get("shared_connector", None)

        if self.shared_connector is not None:
            num_connectors = 1
        else:
            num_connectors = self.injected_hidden_states

        num_layers = end_layer_idx - start_layer_idx
        if self.injected_hidden_states == 1:
            step = self.interaction_step  # default to 1
        else:
            step = (
                num_layers // self.injected_hidden_states
                if num_layers > self.injected_hidden_states
                else 1
            )
        num_interaction_layers = num_layers // step

        if config.get("multihead_connector", False):
            connector_config["number_connector_heads"] = num_interaction_layers

        connector_config["output_hidden_states"] = config.get(
            "output_hidden_states", False
        )
        self.output_hidden_states = config.get("output_hidden_states", False)

        self.extraction_config = config.get("extraction_config", None)

        print(f"extraction_config: {self.extraction_config}")

        self.connector = connector(
            connector_type=connector_type,
            input_dim=vis_dim,
            output_dim=text_dim,
            num_layers=num_connectors,
            connector_config=connector_config,
        )  # nn.ModuleList([nn.Linear(vis_dim, text_dim) for i in range(injected_hidden_states)])

        # Prompt
        self.prompt_tuning = config.get("prompt_tuning", False)
        if self.prompt_tuning:
            prompt_len = config.get("prompt_len", 10)

            prompt_dim = config_opt.word_embed_proj_dim

            mlp = config.get("mlp", True)
            deep = config.get("deep", False)
            nb_prompts = config.get("nb_prompts", 12)
            self.external_input = config.get("external_input", False)

            self.prompt_module = InputPrompts(
                prompt_len=prompt_len,
                prompt_dim=prompt_dim,
                mid_dim=prompt_dim,
                mlp=mlp,
                deep=deep,
                nb_prompts=nb_prompts,
                external_input=self.external_input,
            )

        # Adapters
        self.use_adapters = False

        ## interaction module
        self.cross_modal_module = interaction(
            config,
            input_dim=vis_dim,
            output_dim=text_dim,
            num_layers=num_interaction_layers,
        )

        ## token pruning
        self.vis_tokens = config.get("vis_tokens", "cls")

        self.is_llama = any(
            [k in self.opt_model_name for k in ["llama", "llava", "vicuna"]]
        )

    def get_embed_tokens(
        self,
    ):
        if any([k in self.opt_model_name for k in ["llama", "llava", "vicuna"]]):
            return self.model_text.model.embed_tokens
        else:
            return self.model_text.model.decoder.embed_tokens

    def get_llm_layers(
        self,
    ):
        if any([k in self.opt_model_name for k in ["llama", "llava", "vicuna"]]):
            return self.model_text.model.layers
        elif "opt" in self.opt_model_name:
            return self.model_text.model.decoder.layers
        else:
            raise NotImplemented

    def get_text_model(
        self,
    ):
        if any([k in self.opt_model_name for k in ["llama", "llava", "vicuna"]]):
            return self.model_text.model
        else:
            return self.model_text.model.decoder

    def embed_image(
        self,
        image,
        generation_kwargs=None,
        question=None,
    ):

        vision_states_before = [None] * self.injected_hidden_states
        vision_states_after = [None] * self.injected_hidden_states
        vision_states_connector = [None] * self.injected_hidden_states

        image_embed, image_feat = self.model_vision(image)

        orig_image_feat = list(image_feat)

        image_feat = orig_image_feat[-self.injected_hidden_states :]
        original_image_feat = [[] for i in range(self.injected_hidden_states)]

        for i in range(1, len(image_feat) + 1):
            if self.vis_tokens == "all":
                image_feat_ = image_feat[-i]
            elif self.vis_tokens == "avg":
                image_feat_ = image_feat[-i].mean(1, keepdim=True)
            elif self.extraction_config is not None:
                extraction = self.extraction_config.get("extraction", "avg")
                num_layers = self.extraction_config.get("num_layers", 2)
                if extraction == "avg":
                    image_feat_ = torch.mean(
                        torch.stack(orig_image_feat[-num_layers:], dim=0), dim=0
                    )
                else:
                    raise NotImplemented

            else:
                image_feat_ = image_feat[-i][:, 0, :].unsqueeze(1)

            original_image_feat[-i] = image_feat_

            if self.shared_connector:
                out = self.connector[0](image_feat_)
                if self.output_hidden_states:
                    vision_states_connector[-i] = out[1]
                    image_feat[-i] = out[0]
                else:
                    image_feat[-i] = out
            else:
                image_feat[-i] = self.connector[-i](image_feat_)

            if generation_kwargs is not None and generation_kwargs.get(
                "output_hidden_states", False
            ):
                vision_states_before[-i] = image_feat_
                vision_states_after[-i] = image_feat[-i]

        return (
            image_feat,
            vision_states_before,
            vision_states_after,
            vision_states_connector,
        )

    def prepare_multimodal_input(
        self, input_ids, vis_prefix=None, attention_mask=None, prompt_embeds=None
    ):

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        inputs_embeds = self.get_embed_tokens()(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device
            )

        if vis_prefix is not None:
            ## vis prefix
            prompt_len = 0
            vis_prefix_idx = 0
            vis_pref_ = vis_prefix[vis_prefix_idx]
            if isinstance(vis_pref_, list):
                if isinstance(vis_pref_[0], list):
                    vis_pref = vis_pref_[vis_prefix_idx][0]
                elif len(vis_pref_) == 1:
                    vis_pref = vis_pref_[0]
                else:
                    vis_pref = vis_pref_[0]
            else:
                vis_pref = vis_pref_

            bs_v, bs_t = vis_pref.shape[0], inputs_embeds.shape[0]

            if bs_v != bs_t:
                vis_pref = tile(vis_pref, 0, bs_t // bs_v)

            inputs_embeds = torch.cat((vis_pref, inputs_embeds), dim=1)

            prompt_len += vis_pref.shape[1]

            ## Prompt tuning
            if (
                prompt_embeds is not None
            ):  # and past_key_values_length == 0: # in case of generation don't re add the prompt
                p_embeds = prompt_embeds

                prompt_len += p_embeds.shape[1]
                ## support number of beams > 1, inputs_embeds (nxbs, L, dim)
                if p_embeds.shape[0] < inputs_embeds.shape[0]:
                    p_embeds = p_embeds.repeat(
                        inputs_embeds.shape[0] // p_embeds.shape[0], 1, 1
                    )
                inputs_embeds = torch.cat((p_embeds, inputs_embeds), dim=1)

            if prompt_len > 0:
                attention_mask = F.pad(
                    attention_mask, (prompt_len, 0, 0, 0), "constant", 1
                )

        return inputs_embeds, attention_mask

    def forward(
        self,
        image=None,
        text=None,
        mode="generate",
        return_dict=True,
        labels=None,
        only_image=False,
        only_get_embeddings=False,
        question=None,
        return_time=False,
        noise_modality=False,
        **generation_kwargs,
    ):

        if only_image:
            (
                image_feat,
                vision_states_before,
                vision_states_after,
                vision_states_connector,
            ) = self.embed_image(image, generation_kwargs, question=question)
            text_embeds = self.get_embed_tokens()(text.input_ids)
            attention_mask = text.attention_mask
            return image_feat, text_embeds, attention_mask

        if image is not None:
            (
                image_feat,
                vision_states_before,
                vision_states_after,
                vision_states_connector,
            ) = self.embed_image(image, generation_kwargs, question=question)

            if self.noise_modality or noise_modality:
                image_feat = [
                    torch.rand(feat.shape).to(feat.device) for feat in image_feat
                ]
        else:
            image_feat, vision_states_before, vision_states_after = None, None, None

        if self.prompt_tuning:
            external_input = (
                self.get_embed_tokens()(text.input_ids) if self.external_input else None
            )
            prompts = self.prompt_module.get_prompt(
                text.input_ids.shape[0],
                text.attention_mask.device,
                external_input=external_input,
            )
        else:
            prompts = None

        if self.no_attention_mask:
            attention_mask = None
        else:
            attention_mask = text.attention_mask

        inputs_embeds, attention_mask = self.prepare_multimodal_input(
            text.input_ids,
            vis_prefix=image_feat,
            attention_mask=attention_mask,
            prompt_embeds=prompts,
        )

        if labels is not None:
            # ignore prompt tokens
            src_len, tgt_len = inputs_embeds.shape[1], labels.shape[-1]
            if tgt_len != src_len:
                labels = F.pad(labels, (src_len - tgt_len, 0, 0, 0), "constant", -100)

        if only_get_embeddings:
            return inputs_embeds, attention_mask, labels

        if mode == "train" or mode == "evaluate":

            output_hidden_states = generation_kwargs.get("output_hidden_states", False)
            output_attentions = generation_kwargs.get("output_attentions", False)

            if return_time:
                start = time.time()
                text_output = self.model_text(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=return_dict,
                    labels=labels,
                    output_hidden_states=output_hidden_states,
                    output_attentions=output_attentions,
                )
                end = time.time()
                t = (end - start) * 1000 / inputs_embeds.shape[0]
                return text_output, t
            else:
                text_output = self.model_text(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=return_dict,
                    labels=labels,
                    output_hidden_states=output_hidden_states,
                    output_attentions=output_attentions,
                )

                if generation_kwargs.get("output_hidden_states", False):
                    text_output["vision_states_before"] = vision_states_before
                    text_output["vision_states_after"] = vision_states_after
                    text_output["vision_states_connector"] = vision_states_connector

                return text_output

        elif mode == "generate":

            if return_time:
                start = time.time()
                gen = self.model_text.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )
                end = time.time()
                t = (end - start) * 1000 / inputs_embeds.shape[0]
                if generation_kwargs.get("output_hidden_states", False):
                    gen["vision_states_before"] = vision_states_before
                    gen["vision_states_after"] = vision_states_after
                return gen, t
            else:
                gen = self.model_text.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )

                if generation_kwargs.get("output_hidden_states", False):
                    gen["vision_states_before"] = vision_states_before
                    gen["vision_states_after"] = vision_states_after
                    gen["vision_states_connector"] = vision_states_connector

                return gen
