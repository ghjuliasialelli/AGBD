#!/bin/bash

years=(2019 2020)
modes=("test" "train" "val")

# Get the data for year 2019 and 2020
for year in "${years[@]}"
do

  # Download patches data
  for i in {0..19}
  do
    # Construct the URL with the current value of i
    url="https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/674193/data_subset-${year}-v4_${i}-20.h5"  
    wget "$url"
  done

  # Download tabular data
  for mode in "${modes[@]}"
  do
    wget "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/674193/${mode}_features_${year}-v4.csv"
  done

done

# Download the statistics
wget https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/674193/statistics_subset_2019-2020-v4.pkl

# Download the train/test/val split
wget https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/674193/biomes_splits_to_name.pkl
