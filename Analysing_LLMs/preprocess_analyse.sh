#!/bin/bash

#SBATCH --job-name=compute_inside_vicuna_consec_2_repeat
#SBATCH --nodes=1
#SBATCH --partition=funky
#SBATCH --gpus-per-node=0
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=END,FAIL
###SBATCH --nodelist=zz
#SBATCH --output=/data/mshukor/logs/slurm/compute_inside_vicuna_consec_2_repeat.out
#SBATCH --cpus-per-task=26


cd /home/mshukor/DePALM
source ~/.bashrcs

source activate opt

export LC_ALL=C

export TRANSFORMERS_CACHE=/data/mshukor/data/.cache/transformers

###### Vocabulary distribution
# model=opt
# model=vicuna
# model=llamav2
# model=qformernoptllavafrozen1round
# model=noptllavafrozen1round
# model=llavafrozen1round
# model=llava1round

for model in {vicuna,qformernoptllavafrozen1round,};do
    echo $model
    python compute_histograms.py --model $model --layer_norm --mode consecutives # stats between consecutive layers; kl-distance
    python compute_histograms.py --model $model --layer_norm # stats at the same layer; kl-distance + histograms
done

############################################################# Ious
# model=llava-v1.5-7b
# model=llava_v1_5_baseline_v100
# model=llava_v1_5_baseline_withpt
# model=llava_v1_5_qformer

# python compute_ious_across_layres.py --model $model

############################################################## IMA, cosine, norms

# model=opt
model=vicuna

# model=qformernoptllavafrozen1round
# model=noptllavafrozen1round
# model=llavafrozen1round
# model=llava1round


# model=nopt_ckpts_llavafrozen1round ## for checkpoints

# model=llava


for sim_mode in {avg,median,max,min};do
    echo $sim_mode

    python compute_sim.py --model $model --sim_mode $sim_mode # cosine sim across layers

    python compute_sim.py --model $model --sim_mode $sim_mode --mode inside # cosine sim inside the block
    python compute_sim.py --model $model --sim_mode $sim_mode --mode epochs_imagelayer0 # IMA
    python compute_sim.py --model $model --sim_mode $sim_mode --mode epochs # cosine sim across layers and epochs
    python compute_sim.py --model $model --sim_mode $sim_mode --mode epochs_skipfirst # cosine sim across layers and epochs avoid from scratch
    python compute_sim.py --model $model --sim_mode $sim_mode --mode epochs_skipfirst_imagelayer0 # IMA across layers and epochs avoid from scratch

    ## same as above, allepochs means consider all available checkpoints
    # python compute_sim.py --model $model --sim_mode $sim_mode --mode allepochs_imagelayer0
    # python compute_sim.py --model $model --sim_mode $sim_mode --mode allepochs_skipfirst_imagelayer0
    # python compute_sim.py --model $model --sim_mode $sim_mode --mode allepochs_skipfirst
    # python compute_sim.py --model $model --sim_mode $sim_mode --mode allepochs


    # python compute_sim.py --model $model --sim_mode $sim_mode --mode models # Cosine sim across layers and models
    python compute_sim.py --model $model --sim_mode $sim_mode --mode models_imagelayer0 # IMA across layers and models

    # python compute_sim.py --model $model --sim_mode $sim_mode --mode intra # cosine sim intra modality
done

for sim_mode in {avg,diag_mean,diag_median,median,max};do
    echo $sim_mode
    python compute_sim.py --model $model --sim_mode $sim_mode --mode consecutives # stats between consecutive epochs; norms and cosine sim
    python compute_sim.py --model $model --sim_mode $sim_mode --mode consecutives_inside # same but inside the blocks
done

# ############################################################### Entropy

for model in {vicuna,qformernoptllavafrozen1round};do
    echo $model
    python compute_histograms.py --model $model --layer_norm --mode entropy
done

# ############################################################## tsne

# model=opt
model=vicuna
# model=vicuna_qa

# model=qformernoptllavafrozen1round
# model=noptllavafrozen1round
# model=llavafrozen1round
# model=llava1round

# model=nopt_ckpts_llavafrozen1round ## for checkpoints

python compute_tsne.py --model $model # tsne
# python compute_tsne.py --model $model --mode alltok
# python compute_tsne.py --model $model --mode epochs
python compute_tsne.py --model $model --mode inside
