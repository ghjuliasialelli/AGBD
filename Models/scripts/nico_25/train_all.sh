#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH --output=/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Baseline/logs/training-%A_%a.txt
#SBATCH --error=/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Baseline/logs/training-%A_%a.txt
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=260G
#SBATCH --array=1-5
#SBATCH --job-name=models
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10245MB 

################################################################################################################################
# Establish the paths based on whether we're on the cluster or not

current_directory=$(pwd)
echo "Current Directory: $current_directory"
first_part=$(echo "$current_directory" | cut -d'/' -f2)

if [ "$first_part" == "cluster" ]; then
    echo "Running on a cluster"
    
    # Move the .h5 files
    rsync --include '*v4_*-20.h5' --exclude '*' -aq /cluster/work/igp_psr/gsialelli/Data/patches/ ${TMPDIR}

    # Move the file with the statistics
    #rsync -aq /cluster/work/igp_psr/gsialelli/Data/patches/statistics_subset_2019-v3.pkl ${TMPDIR}/normalization_values_subset.pkl

		rsync -aq /cluster/work/igp_psr/gsialelli/Data/patches/statistics_subset_2019-2020-v4.pkl ${TMPDIR}
    

    # Move the file with the splits
    rsync -aq /cluster/work/igp_psr/gsialelli/Data/AGB/biomes_splits_to_name.pkl ${TMPDIR}
    

elif [ "$first_part" == "scratch" ]; then
    echo "Running on a local machine"
else
    echo "Environment unknown"
fi

# Model parameters ###############################################################################################

if [ "$first_part" == "cluster" ]
then
    JOB_ID=$((SLURM_ARRAY_JOB_ID))
    MODEL_IDX=$((SLURM_ARRAY_TASK_ID))
else
    JOB_ID=0
    MODEL_IDX=0
fi

##########
loss_fn='MSE'
ch="true"
bands=(B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12)
###########

# model itself

arch="nico"
channel_dims=(32 32 64 128 128 128)

# inputs
patch_size=(25 25)
latlon="true"
s1="false"
alos="true"
lc="true"
dem="true"
gedi_dates="false"
s2_dates="false"

# Year to train on
years=(2019 2020)

echo "Year: ${years[@]}"
echo "Architecture: $arch"

# UNet arguments
leaky_relu="false"
max_pool="false"

# training
n_epochs=100000
batch_size=128
limit="false"
reweighting='no' #'no' or 'ifns'
lr=0.001
step_size=30
gamma=0.1
patience=1000
min_delta=0.0
chunk_size=1

# Output path and model name #####################################################################################

model_path=/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Baseline/${arch}



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
norm_strat='pct'

# Launch training ################################################################################################

python3 train.py --model_path $model_path \
                --model_name $model_name \
                --dataset_path $dataset_path \
                --augment "false" \
                --norm "false" \
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
