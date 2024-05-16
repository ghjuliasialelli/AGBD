#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Baseline/logs/eval-%A_%a.txt
#SBATCH --error=/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Baseline/logs/eval-%A_%a.txt
#SBATCH --mem-per-cpu=2G
#SBATCH --tmp=10G
#SBATCH --job-name=eval
#SBATCH --gpus=1

##################################################################################################################
# TO EDIT ########################################################################################################
years=(2019)
arch="fcn"
models=('58217185-1') # '58217185-2' '58217185-3' '58217185-4' '58217185-5')
mode="test"

echo "Years: ${years[@]}"
echo "Mode: $mode"

##################################################################################################################

current_directory=$(pwd)
echo "Current Directory: $current_directory"
first_part=$(echo "$current_directory" | cut -d'/' -f2)

if [ "$first_part" == "cluster" ]; then
    echo "Running on a cluster"
    
    dataset_path=$TMPDIR
    weights_base_path="/cluster/work/igp_psr/tpeters/EcosystemAnalysis/Models/Baseline/"
    plot_folder="/cluster/work/igp_psr/gsialelli/EcosystemAnalysis/Models/Baseline/eval_plots/"

    for model in ${models[@]}
    do
        rsync -aq ${weights_base_path}/${arch}/${model} ${TMPDIR}
    done


    # Move the .h5 files
    rsync --include '*.h5' --exclude '*' -aq /cluster/work/igp_psr/gsialelli/Data/AGB/ ${TMPDIR}

    # Move the file with the statistics
    rsync -aq /cluster/work/igp_psr/gsialelli/Data/AGB/statistics_subset.pkl ${TMPDIR}/normalization_values_subset.pkl

    # Move the file with the splits
    rsync -aq /cluster/work/igp_psr/gsialelli/Data/AGB/biomes_splits_to_name.pkl ${TMPDIR}
    
elif [ "$first_part" == "scratch2" ]; then
    echo "Running on a local machine"
    dataset_path='local'
    plot_folder='/scratch2/gsialelli/EcosystemAnalysis/Models/Baseline/eval_plots/'
    
else
    echo "Environment unknown"
fi


# Launch evaluation ##############################################################################################

python eval.py --dataset_path "$dataset_path" --arch "$arch" --models "${models[@]}" --years "${years[@]}" --plot_folder "$plot_folder" --mode "$mode"