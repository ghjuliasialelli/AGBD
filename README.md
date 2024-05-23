# :evergreen_tree: AGBD: A Global-scale Biomass Dataset :deciduous_tree:
Authors: Ghjulia Sialelli ([gsialelli@ethz.ch](mailto:gsialelli@ethz.ch)), Torben Peters, Jan Wegner, Konrad Schindler

This repository contains the code used to create the results presented in the eponymous paper. We curated a dataset from various remote-sensing data sources (Sentinel-2 L2A, ALOS-2 PALSAR-2 yearly mocaics, JAXA Digital Elevation Model, Copernicus Land Cover, Lang et al.'s Canopy Height Map) and GEDI L4A Above-Ground Biomass (AGB) data. We developed benchmark models for the task of estimating AGB.

### Table of Contents
1. [Downloading the data](https://github.com/ghjuliasialelli/AGBD#Downloading-the-data)
2. [Models training](https://github.com/ghjuliasialelli/AGBD#Models-training)
3. [Patches creation](https://github.com/ghjuliasialelli/AGBD#Patches-creation)

## Downloading the data
The dataset is openly accessible on [HuggingFace](https://huggingface.co/datasets/prs-eth/AGBD), where it is stored in a streamable ML-ready format. However, the patch size is fixed (15x15). Users who want to experiment with a different patch size can download the data as described, and use the provided data loader. Note that the data is 300GB. 

You can either download: the tabular data, that only contains data for the central pixel (this is equivalent to using a patch size of 1x1), in `.csv` format; or the `.h5` data which contains the 25x25 patches. You can download data for the year 2019 and/or the year 2020. 

To download the files with extension `<ext>` (`h5` or `csv`) for the year `<year>` (`2019` or `2020`), run: 
```
wget https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/674193/<year>_<ext>.tar.gz
```

This will download a compressed tarball, which you can uncompress via `tar -xzvf <year>_<ext>.tar.gz`. 


## Models training
Should you wish to reproduce our results, we provide in the [Models section](https://github.com/ghjuliasialelli/AGBD/tree/main/Models) of this repository the code we used to train our benchmark models. You can find further instruction there on how to do it.

## Patches creation
We provide an example for the patches creation procedure, in the [Data section](https://github.com/ghjuliasialelli/AGBD/tree/main/Data) of this repository. Further instructions can be found there.

