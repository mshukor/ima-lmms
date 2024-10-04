#!/bin/bash

#SBATCH --job-name=DePALM_L_audio_caption_l0_1_qsformerl10_ast_test2
#SBATCH --nodes=1
#SBATCH --partition=gpu_p2
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/slurm/DePALM_L_audio_caption_l0_1_qsformerl10_ast_test2.out
###SBATCH --nodelist=jean-zay-ia824
#SBATCH --cpus-per-task=48
###SBATCH --exclusive
#SBATCH --time=1:00:00
#SBATCH --qos=qos_gpu-dev
###SBATCH -C v100-32g
###SBATCH -C a100



cd ~/DePALM
source ~/.bashrc

source activate epalm
rm core-*

export LC_ALL=C

rm core-*
ulimit -c 0 # prevent generating dump core files when crashing

export TRANSFORMERS_CACHE=.cache/huggingface/transformers
export LC_ALL=C

MAIN_PORT=29500
NUM_GPUS=8



data_dir=$SCRATCH/data/audiocaps

config=./configs/audio/DePALM_L_audio_caption_ast.yaml
output_dir=$WORK/logs/DePalm/DePALM_L_audio_caption_llamav2vicuna_ast

# config=./configs/audio/DePALM_L_audio_caption_audiomae.yaml
# output_dir=$WORK/logs/DePalm/DePALM_L_audio_caption__llamav2vicuna_ast_audiomae



accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=$NUM_GPUS --num_machines=1 --main_process_port=$MAIN_PORT accelerate_training/train.py \
--config $config \
--output_dir  $output_dir \
--data_dir $data_dir \
--save_best \
--low_cpu \
--text_model lmsys/vicuna-7b-v1.5 \
--vision_model ast \
--dataset_name audiocaps \


##### To evaluate the model
# --evaluate --checkpoint $output_dir/checkpoint_best.pth \

##### To save the hidden states
# --evaluate --checkpoint $output_dir/checkpoint_best.pth \
# --test_topk 400 --save_hidden_states --inference_mode evaluate --output_intermediate_hidden_states

##### To train with different backbones
# --vision_model ast \
# --vision_model hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m \

# --text_model pretrained_models/llama/llamav2_7b_hf \
# --text_model pretrained_models/llama/llamav1_hf/llama-7b \
# --text_model facebook/opt-6.7b \
# --text_model lmsys/vicuna-7b-v1.5 \
