#!/bin/bash

# Download data for the year 2019
for i in {0..19}
do
  # Construct the URL with the current value of i
  url="https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/674193/data_subset-2019-v4_${i}-20.h5"

  # Use wget to download the file
  wget "$url"
done

# Download data for the year 2019
for i in {0..19}
do
  # Construct the URL with the current value of i
  url="https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/674193/data_subset-2020-v4_${i}-20.h5"

  # Use wget to download the file
  wget "$url"
done

# Download the statistics
wget https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/674193/statistics_subset_2019-2020-v4.pkl

# Download the train/test/val split
wget https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/674193/biomes_splits_to_name.pkl
