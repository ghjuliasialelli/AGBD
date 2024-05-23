## Models training

This folder contains everything needed for the training and/or testing of our models. Note that all models were trained on a cluster with Slurm Workload Manager, however, the bash scripts can also be run locally.


Here is a breakdown of the files and their content:

| Filename | What it does |
|----------|----------|
| [fcn_15](fcn_15) | Bash scripts* for training the FCN model on (15x15) patches. |
| [fcn_25](fcn_25) | Bash scripts* for training the FCN model on (25x25) patches. |
| [nico_15](nico_15) | Bash scripts* for training the Nico et al. model on (15x15) patches. |
| [nico_25](nico_25) | Bash scripts* for training the Nico et al. model on (25x25) patches. |
| [unet_15](unet_15) | Bash scripts* for training the UNet model on (15x15) patches. |
| [unet_25](unet_25) | Bash scripts* for training the UNet model on (25x25) patches. |
| [dataset.py](dataset.py) | Defines the Data Loader. |
| [eval.py](eval.py) | Implements the evaluation of the model(s). |
| [eval.sh](eval.sh) | Launches the evaluation of the model(s). |
| [inference_helper.py](inference_helper.py) | Helper functions for inference. |
| [inference.py](inference.py) | Implements inference code for the model(s). |
| [inference.sh](inference.sh) | Launches inference for the model(s). |
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


*Note: there is a `.sh` file per ablation study.


### Launch training

To launch the training of all of the models and all of their ablation studies, run `python run_jobs.py`. Otherwise, launch the appropriate scripts individually via, for example, `sbatch unet_15/train_all.sh`.

### Pre-trained weights

To download the weigths of the trained models, launch the following command:
```
wget https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/674193/weights.tar.gz
```
This will download a compressed tarball, which you can uncompress via `tar -xzvf weights.tar.gz`.

