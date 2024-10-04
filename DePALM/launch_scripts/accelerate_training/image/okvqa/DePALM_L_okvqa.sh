#!/bin/bash

#SBATCH --job-name=DePALM_okvqa_qformerl10_llamav2vicuna_vitl_save
#SBATCH --nodes=1
#SBATCH --partition=gpu_p2
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/slurm/DePALM_okvqa_qformerl10_llamav2vicuna_vitl_save.out
###SBATCH --nodelist=jean-zay-ia824
#SBATCH --cpus-per-task=12
###SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --qos=qos_gpu-dev
###SBATCH -C v100-32g



cd ~/DePALM
source ~/.bashrc

source activate epalm
rm core-*

export LC_ALL=C

rm core-*

export TRANSFORMERS_CACHE=.cache/huggingface/transformers
export LC_ALL=C

export NCCL_P2P_LEVEL=NVL

process_port=29505

NUM_GPUS=8
data_dir=$SCRATCH/data/okvqa

config=./configs/image/okvqa/DePALM_L_okvqa.yaml
output_dir=$WORK/logs/DePalm/DePALM_okvqa_llamav2vicuna_vitl




accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=$NUM_GPUS --num_machines=1 --main_process_port=$process_port accelerate_training/train.py \
--config $config \
--output_dir  $output_dir \
--data_dir $data_dir \
--save_best \
--low_cpu \
--dataset_name okvqa \
--text_model lmsys/vicuna-7b-v1.5 \
--vision_model vit_large_patch16_224 \
--v2 \


##### To evaluate the model
# --evaluate --checkpoint $output_dir/checkpoint_best.pth \

##### To save the hidden states
# --evaluate --checkpoint $output_dir/checkpoint_best.pth \
# --test_topk 400 --save_hidden_states --inference_mode evaluate --output_intermediate_hidden_states

##### To train with different backbones
# --vision_model clip_large \
# --vision_model vit_large_patch16_224 \
# --vision_model vit_large_patch16_224.mae \

# --text_model pretrained_models/llama/llamav2_7b_hf \
# --text_model pretrained_models/llama/llamav1_hf/llama-7b \
# --text_model facebook/opt-6.7b \
# --text_model lmsys/vicuna-7b-v1.5 \