# Models training

This folder contains everything needed for the training and/or testing of our models. Note that all models were trained on a cluster with Slurm Workload Manager, however, the bash scripts can also be run locally.


Here is a breakdown of the files and their content:

| Filename | What it does |
|----------|----------|
| [scripts](scripts) | Bash scripts for training the models. |
| [dataset.py](dataset.py) | Defines the Data Loader. |
| [download.sh](download.sh) | Bash script to download the dataset. |
| [inference_helper.py](inference_helper.py) | Helper functions for inference. |
| [inference.py](inference.py) | Implements inference code for the model(s). |
| [inference.sh](inference.sh) | Launches inference for the model(s). |
| [inference.txt](inference.txt) | Lists the tile(s) on which to run inference. |
| [loss.py](loss.py) | Defines the losses of the model(s). |
| [models.py](models.py) | Defines the UNet and FCN models. |
| [nico_net.py](nico_net.py) | Defines the model implemented by Lang et al. |
| [parser.py](parser.py) | Defines the parser for most of the files. |
| [rf.py](rf.py) | Defines the Gradient Boosting Decision Tree model. |
| [run_jobs.py](run_jobs.py) | Launches all bash scripts in all subfolders. |
| [table.py](table.py) | Generates the tabular dataset for GBDT, and the corresponding Data Loader. |
| [train.py](train.py) | Defines the training procedure. |
| [train.sh](train.sh) | Launches the training procedure. |
| [wrapper.py](wrapper.py) | Wrapper for the training procedure. |


## Downloading the data
You can simply run the following command, to download the data into a newly created `Data/` folder. :warning:	The data represents 300GB :warning:	
```
bash download.sh
```
This will download the raw `.h5` files (for training the FCN/UNet/Lang et al. models) and `.csv` files (for training the GBDT), along with two other files: `biomes_splits_to_name.pkl` which is a dictionary listing the Sentinel-2 tiles considered for each split (train/test/val), and `statistics_subset_2019-2020-v4.pkl` which holds the normalization values for the dataloader.

## Launch training

To launch the training of all of the models and all of their ablation studies, run `python run_jobs.py`. Otherwise, launch the appropriate scripts individually via, for example, `sbatch scripts/unet_15/train_all.sh`. Note that to launch the training, you will need to log-in to your [Weights and Biases](https://wandb.ai/home) account (by simply running `wandb login`), or modify the code to remove wandb logging.

## Inference

To run inference on an example tile, you first need to download some example data, as follows:
```
wget "https://libdrive.ethz.ch/index.php/apps/files/ajax/download.php?dir=%2F674193&files=inference"
```
This includes a handful of Sentinel-2 L2A products, the ALOS-2 PALSAR-2 yearly mosaic for 2020, the JAXA Digital Elevation Model, the Copernicus Land Cover, and the yearly Canopy Height Map for 2020. We take for this example, the Sentinel-2 tile 30NXM, located in Ghana. 


You also need to download the pre-trained weights of the best model for each architecture, launch the following command:
```
wget "https://libdrive.ethz.ch/index.php/apps/files/ajax/download.php?dir=%2F674193&files=pretrained_weights"
```

You can now run inference as follows:
```
bash inference.sh
```
*Note that you will have to adapt the script to your own needs, in terms of cluster access, paths, resources, etc.*
