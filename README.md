# :evergreen_tree: AGBD: A Global-scale Biomass Dataset :deciduous_tree:
Authors: Ghjulia Sialelli ([gsialelli@ethz.ch](mailto:gsialelli@ethz.ch)), Torben Peters, Jan Wegner, Konrad Schindler

[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This repository contains the code used to create the results presented in the eponymous paper. We curated a dataset from various remote-sensing data sources ([Sentinel-2 L2A](https://sentinels.copernicus.eu/web/sentinel/sentinel-data-access/sentinel-products/sentinel-2-data-products/collection-1-level-2a), ALOS-2 PALSAR-2 [yearly mocaics](https://www.eorc.jaxa.jp/ALOS/en/dataset/fnf_e.htm), JAXA [Digital Elevation Model](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm), Copernicus [Land Cover](https://land.copernicus.eu/en/products/global-dynamic-land-cover/copernicus-global-land-service-land-cover-100m-collection-3-epoch-2019-globe), Lang et al. [Canopy Height Map](https://langnico.github.io/globalcanopyheight/)) and GEDI [L4A](https://daac.ornl.gov/GEDI/guides/GEDI_L4A_AGB_Density_V2_1.html) Above-Ground Biomass (AGB) data. We developed benchmark models for the task of estimating Above-Ground Biomass (AGB).

## Installation
To install the packages required to run this code, you can simply run the following commands, which will create a conda virtual environment called `agbd`. Note that this was designed to be installed on Linux systems. 
1. Create a new environment called `agbd` with PyTorch (or follow the instructions on [pytorch.org](https://pytorch.org/get-started/locally)). (Note that CUDA is not available on MacOS)
```
conda create -n agbd python=3.10.9 pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
```
2. Install all other required packages using the `requirements.yml` file
```
conda env update -n agbd -f requirements.yml
```

## Accessing the dataset 🤗
The dataset is openly accessible on [HuggingFace](https://huggingface.co/datasets/prs-eth/AGBD), where it is stored in a streamable ML-ready format. You can use it as follows:
```
#!pip install datasets
from datasets import load_dataset
dataset = load_dataset("prs-eth/AGBD", streaming=True)["train"] # or test, or val
```

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


