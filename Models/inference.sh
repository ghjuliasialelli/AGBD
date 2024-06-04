#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/cluster/scratch/gsialelli/inference/%A_%a.out
#SBATCH --error=/cluster/scratch/gsialelli/inference/%A_%a.out
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --array=1-1

##################################################################################################################
# Define the model to run inference with
arch="fcn"
model='58217185-1'

# For your information, here are the models for which we provide the weigths:
# Format `arch` `model`
# nico 60688111-1
# unet 59641265-1
# fcn 60310360-1
# rf morning-deluge-23-1

################################################################################################################################
# Define the tiles to run inference on

echo "Q: Did you remember to set the # of job arrays?"
# Hint: you have to set it equal to the number of tiles listed in the LIST_TILES_FILE

echo "Q: And the file from which to read the products?"
# Hint: that's the LIST_TILES_FILE variable

# Set the file from which to read the products
LIST_TILES_FILE='inference.txt'
echo "Will read products from file ${LIST_TILES_FILE}"

################################################################################################################################
# Establish the paths based on whether we're on the cluster or not
# (Adapt the paths to your own needs)

current_directory=$(pwd)
echo "Current Directory: $current_directory"

first_part=$(echo "$current_directory" | cut -d'/' -f2)

# The inference is designed to run on a cluster
if [[ "$first_part" == "cluster" ]]; then
    echo "Running on a cluster"
    LIST_PRODS_FILE="/cluster/work/igp_psr/${LIST_TILES_FILE}"
    saving_dir="/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Baseline/predictions"
# When running locally, only the first product from the list will be processed
elif [[ "$first_part" == "scratch2" ]]; then
    echo "Running on a local machine"
    LIST_PRODS_FILE="/scratch2/${LIST_TILES_FILE}"
    SLURM_ARRAY_TASK_MIN=1
    SLURM_ARRAY_TASK_MAX=1
    SLURM_ARRAY_TASK_ID=1
    saving_dir="/scratch2/gsialelli/EcosystemAnalysis/Models/Baseline/predictions"
else
    echo "Environment unknown"
fi

################################################################################################################################
# Parse the tile names from LIST_PRODS_FILE
readarray -t tile_names < ${LIST_PRODS_FILE}
num_tiles=${#tile_names[@]}

# Check if SLURM_ARRAY_TASK_MIN is 1
if [ "$SLURM_ARRAY_TASK_MIN" -ne 1 ]; then
    echo "Assertion failed: SLURM_ARRAY_TASK_MIN is not equal to 1" >&2
    exit 1
fi

# Check if SLURM_ARRAY_TASK_MAX is equal to the length of the array
if [ "$SLURM_ARRAY_TASK_MAX" -ne "$num_tiles" ]; then
    echo "Assertion failed: SLURM_ARRAY_TASK_MAX is not equal to the length of the array" >&2
    exit 1
fi

# Select the i-th element in the array, where i is the current job number - 1 (SLURM_ARRAY_TASK_ID is 1-indexed)
tile=${tile_names[$SLURM_ARRAY_TASK_ID-1]}
echo "Launching predictions for tile: " ${tile}

################################################################################################################################
# Launch the inference

if [[ "$first_part" == "cluster" ]]; then

    python3 inference.py --models ${models[@]} --arch ${arch} --dataset_path ${TMPDIR} \
            --saving_dir ${saving_dir} --tile_name $tile

else

    python3 inference.py --models ${models[@]} --arch ${arch} --dataset_path local \
            --saving_dir ${saving_dir} --tile_name $tile

fi

