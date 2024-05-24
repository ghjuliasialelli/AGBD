#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/cluster/scratch/gsialelli/inference/%A_%a.out
#SBATCH --error=/cluster/scratch/gsialelli/inference/%A_%a.out
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --array=1-20

##################################################################################################################
# TO EDIT ########################################################################################################
arch="fcn"
models=('58217185-1' '58217185-2' '58217185-3' '58217185-4' '58217185-5')

################################################################################################################################

echo "Q: Did you remember to set the # of job arrays?"
echo "Q: And the file from which to read the products?"
echo

# Set the file from which to read the products
LIST_TILES_FILE='gsialelli/BiomassDatasetCreation/Data/download_Sentinel/Sentinel_Clem_California_Cuba_Paraguay_UnitedRepublicofTanzania_Ghana_Austria_Greece_Nepal_ShaanxiProvince_NewZealand_FrenchGuiana.txt'
echo "Will read products from file ${LIST_TILES_FILE}"

################################################################################################################################
# Establish the paths based on whether we're on the cluster or not

current_directory=$(pwd)
echo "Current Directory: $current_directory"

first_part=$(echo "$current_directory" | cut -d'/' -f2)

if [[ "$first_part" == "cluster" ]]; then
    echo "Running on a cluster"
    LIST_PRODS_FILE="/cluster/work/igp_psr/${LIST_TILES_FILE}"
    rsync -aq /cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Baseline/${arch}/ ${TMPDIR}
    rsync -aq /cluster/work/igp_psr/gsialelli/Data/patches/statistics_subset_2019-2020-v4.pkl ${TMPDIR}
elif [[ "$first_part" == "scratch2" ]]; then
    echo "Running on a local machine"
    LIST_PRODS_FILE="/scratch2/${LIST_TILES_FILE}"
    SLURM_ARRAY_TASK_MIN=1
    SLURM_ARRAY_TASK_MAX=479
    SLURM_ARRAY_TASK_ID=1
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

    rsync -aq /cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Nico/global-canopy-height-model/${tile}/2020/preds_inv_var_mean/${tile}_pred.tif ${TMPDIR}
    rsync -aq /cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Nico/global-canopy-height-model/${tile}/2020/preds_inv_var_mean/${tile}_std.tif ${TMPDIR}

    python3 inference.py --models ${models[@]} --arch ${arch} --dataset_path ${TMPDIR} \
            --saving_dir /cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Baseline/predictions \
            --tile_name $tile

else

    python3 inference.py --models ${models[@]} --arch ${arch} --dataset_path local \
            --saving_dir /scratch2/gsialelli/EcosystemAnalysis/Models/Baseline/predictions \
            --tile_name $tile

fi

