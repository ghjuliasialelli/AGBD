#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/training-%A_%a.txt
#SBATCH --error=logs/training-%A_%a.txt
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=260G
#SBATCH --array=1-5
#SBATCH --job-name=models
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10g

##################################################################################################################
# Establish the paths based on whether we're on the cluster or not

current_directory=$(pwd)
echo "Current Directory: $current_directory"
first_part=$(echo "$current_directory" | cut -d'/' -f2)
user=$(echo "$current_directory" | cut -d'/' -f5)

# Replace this with your wandb project
entity="AGBD_CR"

# Load the necessary files to TMPDIR
if [ "$first_part" == "cluster" ]; then
    echo "Running on a cluster"
    
    # Move the .h5 files
    rsync --include '*v4_*-20.h5' --exclude '*' -aq /cluster/work/igp_psr/gsialelli/Data/patches/ ${TMPDIR}

    # Move the file with the statistics
    rsync -aq /cluster/work/igp_psr/gsialelli/Data/patches/statistics_subset_2019-2020-v4_new.pkl ${TMPDIR}

    # Move the file with the splits
    rsync -aq /cluster/work/igp_psr/gsialelli/Data/AGB/biomes_splits_to_name.pkl ${TMPDIR}

    # Move the file with the embeddings
    rsync -aq /cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Baseline/cat2vec/embeddings_train.csv ${TMPDIR}

# Print on which system we are running
else
    echo "Running on a local machine"
fi

##################################################################################################################
# Run's parameters ###############################################################################################

if [ "$first_part" == "cluster" ]
then
    JOB_ID=$((SLURM_ARRAY_JOB_ID))
    MODEL_IDX=$((SLURM_ARRAY_TASK_ID))
else
    JOB_ID=0
    MODEL_IDX=0
fi

##################################################################################################################
# Model parameters ###############################################################################################

# Loss function
loss_fn='MSE' # can be one of 'MSE' or 'GNLL'

# Architecture
arch="unet"
patch_size=(25 25) # one of (15 15) or (25 25)

# Check that if loss is GNLL, then _gaussian needs to be in arch
if [ "$loss_fn" == "GNLL" ] && [[ "$arch" != *"gaussian"* ]]; then
    echo "If loss function is GNLL, then architecture must be gaussian." 
    exit 1
fi

# Features to include
ch="false"
latlon="true"
alos="false"
lc="false"
dem="true"
bands="all"

if [ "$bands" == 'all' ]
then
    bands=(B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12)
elif [ "$bands" == 'rgbn' ]
then
    bands=(B02 B03 B04 B08)
elif [ "$bands" == 'no' ]
then
    bands=()
else
    echo "Unknown bands"
    exit 1
fi

# Year to train on
years=(2019 2020)

echo "Year: ${years[@]}"
echo "Architecture: $arch"

# UNet arguments
leaky_relu="false"
max_pool="false"

# FCN arguments
channel_dims=(32 32 64 128 128 128)

# Training arguments
n_epochs=10
batch_size=128
limit="false"
reweighting='no'
lr=0.001
step_size=30
gamma=0.1
patience=1000
min_delta=0.0
chunk_size=1
norm_strat='pct'
num_outputs=1

##################################################################################################################
# Output path and model name #####################################################################################

if [ "$first_part" == "cluster" ]
then
    model_path=/cluster/work/igp_psr/${user}/EcosystemAnalysis/Models/CR_Baseline/${arch}
else
    model_path=/scratch2/gsialelli/EcosystemAnalysis/Models/CR_Baseline/${arch}
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

# Define the model name
if [ "$first_part" == "cluster" ]
then
    dataset_path=$TMPDIR
    model_name=${model_path}/${JOB_ID}-${MODEL_IDX}
else
    dataset_path='local'
    model_name=${model_path}/local
fi


# Launch training ################################################################################################

python3 train.py --model_path $model_path \
                --model_name $model_name \
                --dataset_path $dataset_path \
                --arch $arch \
                --model_idx $MODEL_IDX \
                --loss_fn $loss_fn \
                --latlon $latlon \
                --ch $ch \
                --bands ${bands[@]} \
                --in_features $in_features \
                --alos $alos \
                --lc $lc \
                --dem $dem \
                --num_outputs $num_outputs \
                --channel_dims ${channel_dims[@]} \
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
                --years ${years[@]} \
                --entity $entity