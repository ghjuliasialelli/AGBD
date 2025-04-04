# :evergreen_tree: AGBD: A Global-scale Biomass Dataset :deciduous_tree:
Authors: Ghjulia Sialelli ([gsialelli@ethz.ch](mailto:gsialelli@ethz.ch)), Torben Peters, Jan Wegner, Konrad Schindler

[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc] [![arXiv](https://img.shields.io/badge/arXiv-2406.04928-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2406.04928)


This repository contains the code used to create the results presented in the eponymous [paper](https://arxiv.org/abs/2406.04928). We curated a dataset from various remote-sensing data sources ([Sentinel-2 L2A](https://sentinels.copernicus.eu/web/sentinel/sentinel-data-access/sentinel-products/sentinel-2-data-products/collection-1-level-2a), ALOS-2 PALSAR-2 [yearly mocaics](https://www.eorc.jaxa.jp/ALOS/en/dataset/fnf_e.htm), JAXA [Digital Elevation Model](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm), Copernicus [Land Cover](https://land.copernicus.eu/en/products/global-dynamic-land-cover/copernicus-global-land-service-land-cover-100m-collection-3-epoch-2019-globe), Lang et al. [Canopy Height Map](https://langnico.github.io/globalcanopyheight/)) and GEDI [L4A](https://daac.ornl.gov/GEDI/guides/GEDI_L4A_AGB_Density_V2_1.html) Above-Ground Biomass (AGB) data. We developed benchmark models for the task of estimating Above-Ground Biomass (AGB).

---

## Installation
To install the packages required to run this code, you can simply run the following commands, which will create a conda virtual environment called `agbd`. For more details, follow the instructions on [pytorch.org](https://pytorch.org/get-started/locally).

#### For Linux users
```
conda create -n agbd python=3.10.9 pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
conda env update -n agbd -f requirements.yml
conda activate agbd
```

#### For Mac users
```
conda create -n agbd python=3.10.9 pytorch::pytorch torchvision torchaudio -c pytorch 
conda env update -n agbd -f requirements_mac.yml
conda activate agbd
```


## Accessing the dataset 🤗
The dataset is openly accessible on [HuggingFace](https://huggingface.co/datasets/prs-eth/AGBD), where it is stored in a streamable ML-ready format. You can use it as follows:
```
#!pip install datasets
from datasets import load_dataset
dataset = load_dataset('prs-eth/AGBD', trust_remote_code=True, streaming=True)["train"]  # Options: "train", "val", "test"
```

## :arrows_counterclockwise: Data downloading and Models training
Should you wish to reproduce our results, we provide in the [Models section](https://github.com/ghjuliasialelli/AGBD/tree/main/Models) of this repository the code we used to train our benchmark models. Should you want to reproduce our experiments with the data format we used, you can download the data and use the provided data loader. You can find further instruction on how to do it in the dedicated folder. Note that the data is ~300GB.

## Patches creation
We provide an example for the patches creation procedure, in the [Patches section](https://github.com/ghjuliasialelli/AGBD/tree/main/Patches) of this repository. Further instructions can be found there.

## Dense predictions
Our dense predictions for the region covered by the dataset can be downloaded via the following command (it represents ~40GB) :
```
wget "https://libdrive.ethz.ch/index.php/s/VPio6i5UlXTgir0/download?path=%2F&files=predictions" -O predictions.tar
tar -xvf predictions.tar
```
You will get a `.tif` file per Sentinel-2 tile in the regions of interest.
Please note that those dense predictions pre-date the "latitude / longitude bug" (see the [changelog](changelog.md) for more information). As we are currently working on a better model, we do not generate the post-bug predictions, but will directly upload the best ones shortly.

## :arrow_up: Updates 
See the [changelog](changelog.md) for more information about what was updated with each new commit (when relevant).

## :soon: Coming
- [X] Fix `wget` links and `download.sh` file;
- [X] Give access to *all* model weights (i.e. not just the best performing ones);
- [ ] Technical document explaining how each data source was downloaded and processed (with code), for easy inference;
- [ ] Jupyter notebook for how to run inference on new data;

## :unlock: License

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].


[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg


## :handshake: Citing

If you use AGBD in a scientific publication, we encourage you to add the following reference:

``
@article{AGBD,
  doi = {10.48550/ARXIV.2406.04928},
  url = {https://arxiv.org/abs/2406.04928},
  title = {AGBD: A Global-scale Biomass Dataset},
  publisher = {arXiv},
  year = {2024},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
``

**The conference proceedings citation will replace the arxiv preprint citation soon.**

