#!/bin/bash

#SBATCH --job-name=DePALM_L_video_caption_l0_1_qsformerl10_llamav2vicuna_timesformer
#SBATCH --nodes=1
#SBATCH --partition=gpu_p2
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/slurm/DePALM_L_video_caption_l0_1_qsformerl10_llamav2vicuna_timesformer_test2.out
###SBATCH --nodelist=jean-zay-ia824
#SBATCH --cpus-per-task=48
###SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --qos=qos_gpu-dev
###SBATCH -C v100-32g
##SBATCH -C a100



cd ~/DePALM
source ~/.bashrc

source activate epalm
rm core-*

export LC_ALL=C

rm core-*

export TRANSFORMERS_CACHE=.cache/huggingface/transformers
export LC_ALL=C

MAIN_PORT=29508

NUM_GPUS=8
data_dir=$SCRATCH/data/MSRVTT


config=./configs/video/DePALM_L_video_caption_msrvtt.yaml
output_dir=$WORK/logs/DePalm/DePALM_L_video_caption_llamav2vicuna_timesformer




accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=$NUM_GPUS --num_machines=1 --main_process_port=$MAIN_PORT accelerate_training/train.py \
--config $config \
--output_dir  $output_dir \
--data_dir $data_dir \
--save_best \
--low_cpu \
--dataset_name msrvtt \
--text_model lmsys/vicuna-7b-v1.5 \
--vision_model timesformer \
--v2 \


##### To evaluate the model
# --evaluate --checkpoint $output_dir/checkpoint_best.pth \

##### To save the hidden states
# --evaluate --checkpoint $output_dir/checkpoint_best.pth \
# --test_topk 400 --save_hidden_states --inference_mode evaluate --output_intermediate_hidden_states

##### To train with different backbones
# --vision_model xclipl \
# --vision_model timesformer \
# --vision_model videomae \

# --text_model pretrained_models/llama/llamav2_7b_hf \
# --text_model pretrained_models/llama/llamav1_hf/llama-7b \
# --text_model facebook/opt-6.7b \
# --text_model lmsys/vicuna-7b-v1.5 \
