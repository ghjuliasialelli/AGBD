#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Baseline/logs/training-%A_%a.txt
#SBATCH --error=/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Baseline/logs/training-%A_%a.txt
#SBATCH --mem-per-cpu=2G
#SBATCH --tmp=10G
#SBATCH --array=1-5
#SBATCH --job-name=models
#SBATCH --gpus=1
#SBATCH --gres=gpumem:11245MB 

################################################################################################################################
# Establish the paths based on whether we're on the cluster or not

current_directory=$(pwd)
echo "Current Directory: $current_directory"
first_part=$(echo "$current_directory" | cut -d'/' -f2)

if [ "$first_part" == "cluster" ]; then
    echo "Running on a cluster"
    
    # Move the .h5 files
    rsync --include '*.h5' --exclude '*' -aq /cluster/work/igp_psr/gsialelli/Data/AGB/ ${TMPDIR}

    # Move the file(s) with the statistics
    #rsync -aq /cluster/work/igp_psr/gsialelli/Data/AGB/statistics_subset.pkl ${TMPDIR}/normalization_values_subset.pkl
    rsync -aq /cluster/work/igp_psr/gsialelli/Data/patches/statistics_subset_2019-v3.pkl ${TMPDIR}
    rsync -aq /cluster/work/igp_psr/gsialelli/Data/patches/statistics_subset_2020-v3.pkl ${TMPDIR}

    # Move the file with the splits
    rsync -aq /cluster/work/igp_psr/gsialelli/Data/AGB/biomes_splits_to_name.pkl ${TMPDIR}

elif [ "$first_part" == "scratch2" ]; then
    echo "Running on a local machine"
else
    echo "Environment unknown"
fi

if [ "$first_part" == "cluster" ]
then
    JOB_ID=$((SLURM_ARRAY_JOB_ID))
    MODEL_IDX=$((SLURM_ARRAY_TASK_ID))
else
    JOB_ID=0
    MODEL_IDX=0
fi

##################################################################################################################
# To edit ########################################################################################################

# Loss function
loss_fn='MSE'

# Architecture, can be one of the following: 'fcn', 'unet', 'rf', 'nico'
arch="rf"

# Features to include
ch="true"
bands=(B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12) #(B02 B03 B04 B08) #(B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12)
patch_size=(15 15) # (has to be 2k+1, 2k+1)
latlon="true"
s1="false"
alos="true"
lc="true"
dem="true"
gedi_dates="false"
s2_dates="false"

# Year to train on
years=(2019)

echo "Year: ${years[@]}"
echo "Architecture: $arch"

# Model parameters ###############################################################################################

# FCN arguments
channel_dims=(16 32 64 128 128 128)
max_pool="false"

# UNet arguments
leaky_relu="false"

# Training arguments
norm_strat='pct'
n_epochs=100000
batch_size=256
limit="false"
reweighting='no'
lr=0.001
step_size=30
gamma=0.1
patience=1000
min_delta=0.0
chunk_size=1

# Output path and model name #####################################################################################

if [ "$first_part" == "cluster" ]
then
    model_path=/cluster/work/igp_psr/gsialelli/MT/Models/${arch}
else
    model_path=/scratch2/gsialelli/EcosystemAnalysis/Models/Baseline/weights
fi

num_bands=${#bands[@]}
in_features=$((num_bands+2)) # + 2 because always using `lat_1` and `lat_2`
if [ "$latlon" == "true" ]
then 
    in_features=$((in_features+2)) # + 2 because `lon_1` and `lon_2`
fi
if [ "$ch" == "true" ]
then 
    in_features=$((in_features+2)) # + 2 because `ch` and `ch_std`
fi
if [ "$alos" == "true" ]
then 
    in_features=$((in_features+2)) # + 2 because hh and hv
fi
if [ "$lc" == "true" ]
then 
    in_features=$((in_features+3)) # + 3 because lc sin lc cos and lc prob
fi
if [ "$dem" == "true" ]
then 
    in_features=$((in_features+1))
fi

if [ "$first_part" == "cluster" ]
then
    dataset_path=$TMPDIR
    model_name=${model_path}/${JOB_ID}-${MODEL_IDX}
else
    dataset_path='local'
    model_name=${model_path}/local
fi

num_outputs=1

# Launch training ################################################################################################

python train.py --model_path $model_path \
                --model_name $model_name \
                --dataset_path $dataset_path \
                --augment "false" \
                --arch $arch \
                --model_idx $MODEL_IDX \
                --loss_fn $loss_fn \
                --latlon $latlon \
                --ch $ch \
                --bands $(IFS=" " ; echo "${bands[*]}") \
                --in_features $in_features \
                --s1 $s1 \
                --alos $alos \
                --lc $lc \
                --dem $dem \
                --gedi_dates $gedi_dates \
                --s2_dates $s2_dates \
                --num_outputs $num_outputs \
                --channel_dims $(IFS=" " ; echo "${channel_dims[*]}") \
                --downsample "false" \
                --n_epochs $n_epochs \
                --batch_size $batch_size \
                --lr $lr \
                --step_size $step_size \
                --gamma $gamma \
                --patience $patience \
                --min_delta $min_delta \
                --reweighting $reweighting \
                --norm_strat $norm_strat \
                --limit $limit \
                --patch_size ${patch_size[@]} \
                --chunk_size $chunk_size \
                --leaky_relu $leaky_relu \
                --max_pool $max_pool \
                --years ${years[@]}
