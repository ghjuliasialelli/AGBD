# AGBD: A Global-scale Biomass Dataset
Authors: Ghjulia Sialelli, Torben Peters, Jan Wegner, Konrad Schindler

This repository contains the code used to create the results presented in the eponymous paper. We curated a dataset from various remote-sensing data sources (Sentinel-2 L2A, ALOS-2 PALSAR-2 yearly mocaics, JAXA Digital Elevation Model, Copernicus Land Cover, Lang et al.'s Canopy Height Map) and GEDI L4A Above-Ground Biomass (AGB) data. We developed benchmark models for the task of estimating AGB.

## Table of Contents
1. [Patches creation](https://github.com/ghjuliasialelli/AGBD#Patches-creation)
2. [Downloading the data](https://github.com/ghjuliasialelli/AGBD#Downloading-the-data)


## Patches creation
We provide an example for the patches creation procedure, in the [Data section](https://github.com/ghjuliasialelli/AGBD/tree/main/Data) of this repository. Further instructions can be found there.

## Downloading the data
The dataset is openly accessible on [HuggingFace](https://huggingface.co/datasets/prs-eth/AGBD), where it is stored in a streamable machine-learning-ready format. However, the patch size is fixed (15x15). Users who want to experiment with a different patch size can download the data as described, and use the provided data loader.

TODO ADD INSTRUCTIONS

## Models training
Should you wish to reproduce our results, we provide in the [Models section](https://github.com/ghjuliasialelli/AGBD/tree/main/Models) of this repository the code we used to train our benchmark models. You can find further instruction there on how to do it.


