

<p align="center">
    <br>
    <img src="docs/logo.jpg" width="100" />
    <br>
<p>


<p align="center">
        &nbsp<a href="https://ima-lmms.github.io/">Project Page</a> &nbsp | &nbsp<a href="https://arxiv.org/abs/2405.16700">Paper </a>&nbsp 
</p>

# (IMA) Implicit Multimodal Alignment: On the Generalization of Frozen LLMs to Multimodal Inputs


<p align="center">
    <br>
    <img src="docs/main.jpg" width="700" />
    <br>
<p>

# News
* **[2024.10.12]**: Paper on Skipping Computations in Multimodal LLMs accepted at NeurIPS 2024 RBFM Workshop.
* **[2024.10.04]**: The code is released. 
* **[2024.09.25]**: The IMA paper is accepted at NeurIPS 2024. 
* **[2024.05.27]**: The code will be released soon. 

# Overview

This repo contains scripts to analyse LLMs when exposed to multimodal data. In addition, we provide scripts related to the implications to some of our findings, such as pruning and skipping computations.

Specifically, we provide the implementation to reproduce the following papers:

* NeurIPS 2024: [Implicit Multimodal Alignment: On the Generalization of Frozen LLMs to Multimodal Inputs](https://arxiv.org/abs/2405.16700)
* NeurIPS 2024 RBFM Workshop: [Skipping Computations in Multimodal LLMs](https://arxiv.org/abs/2410.09454)


# Citation
If you found this repository useful, you can cite it as:

```
@article{shukor2024implicit,
  title={Implicit Multimodal Alignment: On the Generalization of Frozen LLMs to Multimodal Inputs},
  author={Shukor, Mustafa and Cord, Matthieu},
  journal={arXiv preprint arXiv:2405.16700},
  year={2024}
}
@article{shukor2024skipping,
  title={Skipping Computations in Multimodal LLMs},
  author={Shukor, Mustafa and Cord, Matthieu},
  journal={arXiv preprint arXiv:2410.09454},
  year={2024}
}
```



## Abstracts

### (IMA) Implicit Multimodal Alignment: On the Generalization of Frozen LLMs to Multimodal Inputs

> Large Language Models (LLMs) have demonstrated impressive performance on multimodal tasks, without any multimodal finetuning. They are the de facto building block for Large Multimodal Models (LMMs), yet, we still lack a proper understanding of their success. In this work, we expose frozen LLMs to image, video, audio and text inputs and analyse their internal representation aiming to understand their generalization beyond textual inputs.

> <strong> Findings.</strong> Perceptual tokens (1) are easily distinguishable from textual ones inside LLMs, with significantly different representations (e.g. live in different narrow cones), and complete translation to textual tokens does not exist. Yet, (2) both perceptual and textual tokens activate similar LLM weights. Despite being different, (3) perceptual and textual tokens are implicitly aligned inside LLMs, we call this the implicit multimodal alignment (IMA), and argue that this is linked to architectural design, helping LLMs to generalize. This provide more evidence to believe that the generalization of LLMs to multimodal inputs is mainly due to their architecture.  

> <strong> Implications. </strong>  (1) We find a positive correlation between the implicit alignment score and the task performance, suggesting that this could act as a proxy metric for model evaluation and selection. (2) A negative correlation exists regarding hallucinations (e.g. describing non-existing objects in images), revealing that this problem is mainly due to misalignment between the internal perceptual and textual representations. (3) Perceptual tokens change slightly throughout the model, thus, we propose different approaches to skip computations (e.g. in FFN layers), and significantly reduce the inference cost. (4) Due to the slowly changing embeddings across layers, and the high overlap between textual and multimodal activated weights, we compress LLMs by keeping only 1 subnetwork (called alpha-SubNet) that works well across a wide range of multimodal tasks. The code will be available here: .




# Experiments

We provide the implementation to reproduce the main results in the paper. We focus on single task finetuning following the [DePALM](https://github.com/facebookresearch/DePALM) setup. 

## Installation

Installation instructions can be found in `docs/installation.md`


## Data

You can download the data from their original websites. Some datasets can be downloaded similar to [eP-ALM](https://github.com/mshukor/DePALM/blob/main/docs/datasets.md).
These datasets contain annotation coming from several public datasets and their use is bounded to their corresponding licenses.


### Training



To train a model for image captioning on COCO, you can use the following:

```
MAIN_PORT=29504
NUM_GPUS=8
CONFIG=config.yaml # path to model config file
OUTPUT_DIR=results/ # path to output logs
TEXT_MODEL=lmsys/vicuna-7b-v1.5 # name of the LLM
VISION_MODEL=vit_large_patch16_224 # name of the vision encoder
DATASET_NAME=coco # name of the dataset


accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=$NUM_GPUS --num_machines=1 --main_process_port=$MAIN_PORT accelerate_training/train.py \
--config $CONFIG \
--output_dir  $OUTPUT_DIR \
--data_dir $DATA_DIR \
--save_best \ # save best checkpoint
--low_cpu \
--text_model $TEXT_MODEL \
--vision_model $VISION_MODEL \
--dataset_name $DATASET_NAME
```


In `DePALM/configs`, we provide the configs files to train on several multimodal dataset, using different LLMs and encoders.

### Pretrained models

You can download some pretrained models from [here](https://huggingface.co/mshukor/IMA-DePALM/tree/main).

The configs to reproduce the training of each model can be found in the same dir as the checkpoint (e.g. ./llamav2vicuna/DePALM_L_clothoaqa_audio_qa_l0_1_qsformerl10_llamav2vicuna_ast/config.yaml)


| LLM         | Encoder     | Dataset         | Train Script | Config File                                                   | Checkpoint Path                                                     |
|-------------|-------------|-----------------|--------------|---------------------------------------------------------------|---------------------------------------------------------------------|
| Vicuna-1.5   | AST      | Clotho AQA      | launch_scripts/accelerate_training/audio/DePALM_L_clothoaqa_audio_qa.sh   | configs/audio/DePALM_L_clothoaqa_audio_qa.yaml | ./llamav2vicuna/DePALM_L_clothoaqa_audio_qa_l0_1_qsformerl10_llamav2vicuna_ast/checkpoint_best.pth |
| Vicuna-1.5   | ViT-L      | COCO Caption        | launch_scripts/accelerate_training/image/caption/DePALM_L_caption.sh   | configs/image/caption/DePALM_L_caption.yaml   | ./llamav2vicuna/DePALM_caption_qformerl10_llamav2vicuna_vitl/checkpoint_best.pth |
| Vicuna-1.5   | ViT-L      | OKVQA           | launch_scripts/accelerate_training/image/okvqa/DePALM_L_okvqa.sh   | configs/image/okvqa/DePALM_L_okvqa.yaml    | ./llamav2vicuna/DePALM_okvqa_qformerl10_llamav2vicuna_vitl/checkpoint_best.pth  |
| Vicuna-1.5   | ViT-L      | VQAv2            | launch_scripts/accelerate_training/image/vqa/DePALM_L_vqa.sh  | configs/image/vqa/DePALM_L_vqa.yaml     | ./llamav2vicuna/DePALM_vqa_qformerl10_llamav2vicuna_vitl/checkpoint_best.pth    |
| Vicuna-1.5   | TimeSformer      | MSRVTT   | launch_scripts/accelerate_training/video/DePALM_L_video_caption_msrvtt.sh   | configs/video/DePALM_L_video_caption_msrvtt.yaml | ./llamav2vicuna/DePALM_L_video_caption_l0_1_qsformerl10_llamav2vicuna_timesformer/checkpoint_best.pth |
| Vicuna-1.5   | ViT-L      | GQA             | launch_scripts/accelerate_training/image/gqa/DePALM_L_gqa.sh  | configs/image/gqa/DePALM_L_gqa.yaml      | ./llamav2vicuna/DePALM_gqa_qformerl10_llamav2vicuna_vitl/checkpoint_best.pth    |
| Vicuna-1.5   | TimeSformer      | Video QA (MSVD) | launch_scripts/accelerate_training/video/DePALM_L_video_qa_msvd.sh   | configs/video/DePALM_L_video_qa_msvd.yaml | ./llamav2vicuna/DePALM_L_video_qa_msvd_l0_1_qsformerl10_llamav2vicuna_timesformer/checkpoint_best.pth |
| Vicuna-1.5   | AST      | AudioCaps   | launch_scripts/accelerate_training/audio/DePALM_L_audio_caption.sh   | configs/audio/DePALM_L_audio_caption_ast.yaml | ./llamav2vicuna/DePALM_L_audio_caption_l0_1_qsformerl10_llamav2vicuna_ast/checkpoint_best.pth |
| OPT         | Timesformer | Video QA (MSVD) | launch_scripts/accelerate_training/video/DePALM_L_video_qa_msvd.sh   | configs/video/DePALM_L_video_qa_msvd.yaml| ./opt/DePALM_L_video_qa_msvd_l0_1_qsformerl10_timesformer/checkpoint_best.pth |
| OPT         | ViT-L        | OKVQA           | launch_scripts/accelerate_training/image/okvqa/DePALM_L_okvqa.sh  | configs/image/okvqa/DePALM_L_okvqa.yaml                 | ./opt/DePALM_okvqa_qformerl10_vitl/checkpoint_best.pth              |
| OPT         | ViT-L        | VQAv2             | launch_scripts/accelerate_training/image/vqa/DePALM_L_vqa.sh  | configs/image/vqa/DePALM_L_vqa.yaml                   | ./opt/DePALM_vqa_qformerl10_vitl/checkpoint_best.pth                |
| OPT         | XCLIP L     | Video QA (MSVD) | launch_scripts/accelerate_training/video/DePALM_L_video_qa_msvd.sh  | configs/video/DePALM_L_video_qa_msvd.yaml | ./opt/DePALM_L_video_qa_msvd_l0_1_qsformerl10_xclipl/checkpoint_best.pth |
| OPT         | CLIP-L      | VQAv2             | launch_scripts/accelerate_training/image/vqa/DePALM_L_vqa.sh  | configs/image/vqa/DePALM_L_vqa.yaml          | ./opt/DePALM_L_vqa_l0_1_qsformerl10_clipl/checkpoint_best.pth       |
| OPT         | CLIP-L      | OKVQA           | launch_scripts/accelerate_training/image/okvqa/DePALM_L_okvqa.sh  | configs/image/okvqa/DePALM_L_okvqa.yaml                | ./opt/DePALM_okvqa_qformerl10_clipl/checkpoint_best.pth             |
| OPT         | Timesformer | MSRVTT   | launch_scripts/accelerate_training/video/DePALM_L_video_caption_msrvtt.sh  | configs/video/DePALM_L_video_caption_msrvtt.yaml | ./opt/DePALM_L_video_caption_l0_1_qsformerl10_timesformer/checkpoint_best.pth |
| OPT         | ViT-L        | COCO Caption         | launch_scripts/accelerate_training/image/caption/DePALM_L_caption.sh  | ./opt/DePALM_caption_qformerl10_vitl/config.yaml               | ./opt/DePALM_caption_qformerl10_vitl/checkpoint_best.pth            |
| OPT         | XCLIP-L     | Video QA (MSRVT) | launch_scripts/accelerate_training/video/DePALM_L_video_qa_msrvtt.sh  | .configs/video/DePALM_L_video_qa_msrvtt.yaml | ./opt/DePALM_L_video_qa_msrvtqa_l0_1_qsformerl10_xclipl/checkpoint_best.pth |
| OPT         | AST         | Clotho Caption  | launch_scripts/accelerate_training/audio/DePALM_L_clotho_audio_caption.sh  | configs/audio/DePALM_L_clotho_audio_caption_ast.yaml | ./opt/DePALM_L_clotho_audio_caption_l0_1_qsformerl10_ast/checkpoint_best.pth |
| OPT         | CLIP-L      | COCO Caption         | launch_scripts/accelerate_training/image/caption/DePALM_L_caption.sh  | configs/image/caption/DePALM_L_caption.yaml      | ./opt/DePALM_L_caption_l0_1_qsformerl10_clipl/checkpoint_best.pth   |
| OPT         | CLIP-L      | GQA             | launch_scripts/accelerate_training/image/gqa/DePALM_L_gqa.sh  | configs/image/gqa/DePALM_L_gqa.yaml                  | ./opt/DePALM_gqa_qformerl10_clipl/checkpoint_best.pth               |
| OPT         | AST         | Clotho AQA      | launch_scripts/accelerate_training/audio/DePALM_L_clothoaqa_audio_qa.sh  | configs/audio/DePALM_L_clothoaqa_audio_qa.yaml | ./opt/DePALM_L_clothoaqa_audio_qa_l0_1_qsformerl10_ast/checkpoint_best.pth |
| OPT         | ViT-L        | GQA             | launch_scripts/accelerate_training/image/gqa/DePALM_L_gqa.sh  | configs/image/gqa/DePALM_L_gqa.yaml                   | ./opt/DePALM_gqa_qformerl10_vitl/checkpoint_best.pth                |
| OPT         | AST         | AudioCaps   | launch_scripts/accelerate_training/audio/DePALM_L_audio_caption.sh | configs/audio/DePALM_L_audio_caption_ast.yaml  | ./opt/DePALM_L_audio_caption_l0_1_qsformerl10_ast/checkpoint_best.pth |
| OPT         | Timesformer | Video QA (MSRVT) | launch_scripts/accelerate_training/video/DePALM_L_video_qa_msrvtt.sh  | configs/video/DePALM_L_video_qa_msrvtt.yaml | ./opt/DePALM_L_video_qa_msrvtqa_l0_1_qsformerl10_timesformer/checkpoint_best.pth |
| OPT         | XCLIP L     | MSRVTT   | launch_scripts/accelerate_training/video/DePALM_L_video_caption_msrvtt.sh  | configs/video/DePALM_L_video_caption_msrvtt.yaml | ./opt/DePALM_L_video_caption_l0_1_qsformerl10_xclipl/checkpoint_best.pth |


## Analysis

We provide the scripts to reproduce the main results in the IMA paper:

```
compute_histograms.py # compute the histogram for tokens/vocab distributions, entropy and kl distance
compute_ious_across_layres.py # Compute IoUs plots between pruning masks
compute_sim.py # compute different similarity measure between multimodal tokens
compute_tsne.py # Compute tsne maps for multimodal tokens across layers
```

Some examples can be found in `Analysing_LLMs/preprocess_analyse.sh`


In addition, to analyse pruning masks:
```
merge_results_csv.py # organize results in .xlsx file
compute_baseline_mask.py # applying some operations on prunning masks (e.g. intersection)
compute_ious.py # compute intersection over union between pruning masks
```

## Efficiency and model compression

### Skipping computation

The following command can be used to skip blocks/layers inside LLMs:
```
skip_mode=baseline # no skipping

skip_mode=skip_block # skip entire blocks
skip_mode=context_skip_block # skip entire blocks (including the visual/textual prompt)
skip_mode=onlycontext_skip_block # skip entire blocks (only the visual/textual prompt)


skip_mode=skip_ffn # skip ffns
skip_mode=context_skip_ffn # skip ffns (including the visual/textual prompt)
skip_mode=onlycontext_skip_ffn # skip ffns (only the visual/textual prompt)


skip_mode=parrallel_attn # parrallelizing SA/FFNs
skip_mode=context_parrallel_attn  # parrallelizing SA/FFNs (including the visual/textual prompt)


skip_interval=2 # skip each 2 blocks
start_layer=0 # start skipping from layer 0


accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=$NUM_GPUS --num_machines=1 --main_process_port=29503 accelerate_training/train.py \
--config $config \
--output_dir  $output_dir \
--data_dir $data_dir \
--save_best \
--low_cpu \
--text_model $text_model \
--vision_model $vision_model \
--v2 \
--dataset_name $dataset_name \
--test_split $test_split \
--skipping_mode \
--skip_mode $skip_mode \
--skip_interval $skip_interval \
--start_layer $start_layer \
--output_log_name v3_skipping_${skip_mode}_sl${start_layer}.csv \
--evaluate --checkpoint $output_dir/checkpoint_best.pth
```

### Pruning

To prune LLMs weights using Wanda or Magnitude pruning, you can use the following:
```
sparsity_ratio=0.3 # prune 30% of model parameters
num_calibration_data=256 # number of calibration examples

prune_method=wanda # Wanda pruning
prune_method=magnitude # Magnitude pruning
prune_method=given_mask # Pruning given a pruning mask

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=$NUM_GPUS --num_machines=1 --main_process_port=29503 accelerate_training/train.py \
--config $config \
--output_dir  $output_dir \
--data_dir $data_dir \
--save_best \
--low_cpu \
--text_model facebook/opt-6.7b \
--vision_model clip_large \
--v2 \
--test_split karpathy_val \
--num_calibration_data $num_calibration_data \
--sparsity_ratio $sparsity_ratio \
--sparsity_type unstructured \
--prune_method $prune_method \
--evaluate --checkpoint $output_dir/checkpoint_best.pth \
--output_log_name wandasparsitywithans_test_log.csv \
--with_answers
```

