# :evergreen_tree: AGBD: A Global-scale Biomass Dataset :deciduous_tree:
Authors: Ghjulia Sialelli ([gsialelli@ethz.ch](mailto:gsialelli@ethz.ch)), Torben Peters, Jan Wegner, Konrad Schindler

[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This repository contains the code used to create the results presented in the eponymous paper. We curated a dataset from various remote-sensing data sources (Sentinel-2 L2A, ALOS-2 PALSAR-2 yearly mocaics, JAXA Digital Elevation Model, Copernicus Land Cover, Lang et al.'s Canopy Height Map) and GEDI L4A Above-Ground Biomass (AGB) data. We developed benchmark models for the task of estimating AGB.

## Accessing the dataset
The dataset is openly accessible on [HuggingFace](https://huggingface.co/datasets/prs-eth/AGBD), where it is stored in a streamable ML-ready format. 

## Data downloading and Models training
Should you wish to reproduce our results, we provide in the [Models section](https://github.com/ghjuliasialelli/AGBD/tree/main/Models) of this repository the code we used to train our benchmark models. Should you want to reproduce our experiments with the data format we used, you can download the data and use the provided data loader. You can find further instruction on how to do it in the dedicated folder. Note that the data is ~300GB.

## Patches creation
We provide an example for the patches creation procedure, in the [Patches section](https://github.com/ghjuliasialelli/AGBD/tree/main/Patches) of this repository. Further instructions can be found there.

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].


[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg


