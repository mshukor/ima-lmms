#!/bin/bash

#SBATCH --job-name=DePALM_L_audio_caption_l0_1_qsformerl10_ast_repair_fromskip2
#SBATCH --nodes=1
#SBATCH --partition=gpu_p2
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/slurm/DePALM_L_audio_caption_l0_1_qsformerl10_ast_repair_fromskip2.out
###SBATCH --nodelist=jean-zay-ia824
#SBATCH --cpus-per-task=48
###SBATCH --exclusive
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
###SBATCH -C v100-32g



cd ~/DePALM
source ~/.bashrc

source activate epalm
rm core-*

export LC_ALL=C

rm core-*

export TRANSFORMERS_CACHE=.cache/huggingface/transformers
export LC_ALL=C


NUM_GPUS=8
config=./configs/audio/DePALM_L_audio_caption_astm.yaml
data_dir=$SCRATCH/data/audiocaps
# output_dir=$WORK/logs/DePalm/DePALM_L_audio_caption_l0_1_qsformerl10_ast_repair_fromallwithans
# output_dir=$WORK/logs/DePalm/DePALM_L_audio_caption_l0_1_qsformerl10_ast
output_dir=$WORK/logs/DePalm/DePALM_L_audio_caption_l0_1_qsformerl10_ast_repair_fromskip2
# output_dir=$WORK/logs/DePalm/DePALM_L_audio_caption_l0_1_qsformerl10_ast_opt2_7b



# ePALM_L_audio_caption_l0_1_qsformerl10_llamav2_ast
# ePALM_L_audio_caption_l0_1_qsformerl10_ast


# ################# Pruning
# sparsity_ratio=0.5
# ex_sparsity=0.7

# # mask_dir=$WORK
# mask_dir=$STORE


# # # from all
# # mask_path=$mask_dir/logs/DePalm/masks/masks_onlypromptwithans_all_masks_W_masks_s0_5.pth

# # # from image
# # mask_path=$mask_dir/logs/DePalm/masks/masks_onlypromptwithans_all_image_masks_W_masks_s0_5.pth

# # ## from all 0.3
# # mask_path=$mask_dir/logs/DePalm/masks/masks_onlypromptwithans_all_masks_W_masks_s0_3.pth

# # # from image 0.3
# # mask_path=$mask_dir/logs/DePalm/masks/masks_onlypromptwithans_all_image_masks_W_masks_s0_3.pth

# # ### from vqa v2
# # mask_dir=$WORK/logs/DePalm
# # mask_path=$mask_dir/DePALM_L_vqa_l0_1_qsformerl10_clipl/masks_withans/W_masks_s0.5.pth

# # mask_dir=$WORK/logs/DePalm
# # mask_path=random_mask
# # sparsity_ratio=0.7
# # ex_sparsity=0.7


# # # from all image and text
# # mask_path=$mask_dir/logs/DePalm/masks/all_masks_from_path_s0_5.pth


# # # from all withans
# # mask_path=$mask_dir/logs/DePalm/masks/masks_withans_all_masks_W_masks_s0_5.pth

# # # from all withans 0.3
# # mask_path=$mask_dir/logs/DePalm/masks/masks_withans_all_masks_W_masks_s0_3.pth


# # # from image withans
# # mask_path=$mask_dir/logs/DePalm/masks/masks_withans_all_image_masks_W_masks_s0_5.pth

# mask_dir=$WORK/logs/DePalm
# mask_path=random_mask
# sparsity_ratio=0.47
# ex_sparsity=0.47



# # # # config=$WORK/logs/DePalm/DePALM_L_vqa_l0_1_qsformerl10_clipl/config.yaml


# accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=$NUM_GPUS --num_machines=1 --main_process_port=29504 accelerate_training/train.py \
# --config $config \
# --output_dir  $output_dir \
# --data_dir $data_dir \
# --save_best \
# --low_cpu \
# --text_model facebook/opt-6.7b \
# --vision_model ast \
# --dataset_name audiocaps \
# --v2 \
# --sparsity_ratio $sparsity_ratio \
# --sparsity_type unstructured \
# --prune_method given_mask \
# --mask_path $mask_path \
# --ex_sparsity $ex_sparsity \
# --evaluate --checkpoint $output_dir/checkpoint_best.pth


# # --evaluate --checkpoint $output_dir/checkpoint_best.pth




########################### Skipping
# skip_mode=fixed_skip_block
skip_mode=context_fixed_skip_block
skip_interval=2

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=$NUM_GPUS --num_machines=1 --main_process_port=29504 accelerate_training/train.py \
--config $config \
--output_dir  $output_dir \
--data_dir $data_dir \
--save_best \
--low_cpu \
--text_model facebook/opt-6.7b \
--vision_model ast \
--dataset_name audiocaps \
--v2 \
--skipping_mode \
--skip_mode $skip_mode \
--skip_interval $skip_interval





# #######################3 Normal


# vision_model=ast
# text_model=facebook/opt-2.7b
# # text_model=facebook/opt-1.3b
# # text_model=facebook/opt-350m

# accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=$NUM_GPUS --num_machines=1 --main_process_port=29503 accelerate_training/train.py \
# --config $config \
# --output_dir  $output_dir \
# --data_dir $data_dir \
# --save_best \
# --low_cpu \
# --text_model $text_model \
# --vision_model $vision_model \
# --dataset_name audiocaps \
# --v2 \

# # --resume --checkpoint $output_dir/checkpoint_last.pth
