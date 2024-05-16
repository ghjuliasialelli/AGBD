"""


"""

############################################################################################################################
# Imports

import os
from table import RF_GEDIDataset
from models import Net
from wrapper import Model
from dataset import GEDIDataset
from torch.utils.data import DataLoader
from torch import set_float32_matmul_precision
import wandb
from os.path import join
import lightgbm as lgb
import matplotlib.pyplot as plt
from os.path import isdir
from os import mkdir
from matplotlib.colors import LogNorm
import argparse
from tqdm import tqdm
import numpy as np
import torch

############################################################################################################################
# Execute


def eval_parser() :

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type = str, required = True, help = 'Path to the dataset')
    parser.add_argument('--arch', type = str, required = True, help = 'Architecture of the model')
    parser.add_argument('--models', type = str, nargs = '+', required = True, help = 'Model names')
    parser.add_argument('--years', type = int, nargs = '+', help = 'Year of the dataset')
    parser.add_argument('--plot_folder', type = str, required = True)
    parser.add_argument('--mode', required = True, type = str, help = 'Mode of the dataset')
    args = parser.parse_args()

    return args, args.dataset_path, args.arch, args.models, args.years, args.plot_folder, args.mode


def get_mapping(api, arch) :
    """
    {'58742596-1': 'kpq024cl', '58742596-2': 'l7gnvpae'}
    """
    runs = api.runs(f"gs-tp-biomass/{arch}")
    run_mapping = {}
    for run in runs: run_mapping[run.name] = run.path[-1]
    return run_mapping


def get_state_dict(path, arch, model_name):
    """
        Content: load a Pytorch model state dict. Some processing has to be done because keys sometimes do not match,
        because of the way Pytorch flattens the names of the layers.

        input:
        - `path` (str) : path to the checkpoint;
        - `model_name` (str) : name of the model.

        output (dict) : the state dict of the checkpoint.
    """

    state_dict = torch.load(join(path, arch, f'{model_name}_best.ckpt'), map_location=torch.device('cpu'))['state_dict'] # TODO maybe remove map_location
    rm_prefix = 'model.model.'
    state_dict = {k[len(rm_prefix):] if k.startswith(rm_prefix) else k:v for k,v in state_dict.items()}
    return state_dict


if __name__ == '__main__' :

    # Parse the arguments
    args, dataset_path, arch, models, years, plot_folder, mode = eval_parser()

    # Settings
    set_float32_matmul_precision('high')
    if (dataset_path == 'local') : accelerator, cpus_per_task = 'auto', 8
    else: accelerator, cpus_per_task = 'gpu', int(os.environ.get('SLURM_CPUS_PER_TASK'))
    if cpus_per_task is None: cpus_per_task = 16

    # Define the local dataset paths
    local_dataset_paths = {'h5':'/scratch2/gsialelli/patches', 
                        'norm': f'/scratch2/gsialelli/patches', 
                        'map': '/scratch2/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/biomes_split',
                        'ckpt': '/scratch2/gsialelli/EcosystemAnalysis/Models/Baseline/weights'}

    if dataset_path == 'local' : 
        dataset_path = local_dataset_paths
        debug = True
    else: 
        dataset_path = {k: dataset_path for k in local_dataset_paths.keys()}
        debug = False

    # We get the config for one of the models
    api = wandb.Api()
    wandb_mapping = get_mapping(api, arch)
    wandb_name = wandb_mapping[models[0]]
    cfg = api.run(f'gs-tp-biomass/{arch}/{wandb_name}').config
    for key, value in cfg.items(): setattr(args, key, value)

    # Load the dataset
    if arch == 'rf': 
        test_dataset = RF_GEDIDataset(data_path = dataset_path['h5'], mode = mode, args = args, years = years)
    else:
        test_dataset = GEDIDataset(paths = dataset_path, years = years, chunk_size = 1, mode = mode, args = args, debug = debug)
        test_loader = DataLoader(test_dataset, batch_size = 512, shuffle = False, num_workers = cpus_per_task, pin_memory = True)

    models_rmses = []
    models_preds = []
    for i, model_name in enumerate(models):

        model_preds, model_labels = [], []

        # Build the network based on the architecture requested
        if arch in ['fcn', 'fcn_gaussian', 'unet', 'nico']:            

            # Initialize the model
            model = Net(model_name = arch, in_features = args.in_features, num_outputs = args.num_outputs, 
                        channel_dims = args.channel_dims, max_pool = args.max_pool, downsample = None,
                        leaky_relu = args.leaky_relu, patch_size = args.patch_size)
            
            model = Model(model, lr = args.lr, step_size = args.step_size, gamma = args.gamma, 
                            patch_size = args.patch_size, downsample = args.downsample, 
                            loss_fn = args.loss_fn)
        
            state_dict = state_dict = torch.load(join(dataset_path['ckpt'], arch, f'{model_name}_best.ckpt'), map_location = torch.device('cpu'))['state_dict'] # TODO maybe remove map_location 
            # get_state_dict(dataset_path['ckpt'], arch, model_name)
            model.load_state_dict(state_dict) 
            #model.to(0)
            model.eval()
            model.model.eval()
            model = model.model

            for i, (images, labels) in enumerate(tqdm(test_loader)) :

                if i == 100 : break
                
                center = cfg['patch_size'][0] // 2
                #predictions = model(images.to(0)).cpu().detach().numpy()
                predictions = model(images).cpu().detach().numpy()

                model_preds.append(predictions[:, 0, center, center])
                model_labels.append(labels)


        elif args.arch == 'rf':
            model = lgb.Booster(model_file = join(dataset_path['ckpt'], f'{model_name}.txt'))
            test_preds = model.predict(test_dataset.features)
            model_preds.append(test_preds)
            test_labels = test_dataset.labels['agbd'].to_numpy()
            model_labels.append(test_labels)
            
        # Now calculate the test RMSE for this model
        test_preds = np.concatenate(model_preds)
        test_labels = np.concatenate(model_labels)
        test_rmse = np.sqrt(np.mean(np.power(test_preds - test_labels, 2)))
        print(f'> Model #{i+1} {mode} RMSE: {test_rmse:.2f}')
        models_rmses.append(test_rmse)
        models_preds.append(test_preds)
    
    mean_rmse = np.mean(models_rmses)
    std_rmse = np.std(models_rmses)
    print(f'Ensemble test RMSE: {mean_rmse:.2f} +/- {std_rmse:.2f}')

    # Preparations to save plots
    if not isdir(plot_folder): mkdir(plot_folder)
    output_file_name = f'{arch}_{models[0].split('-')[0]}_{'-'.join([str(year) for year in years])}'
    labels = test_labels
    preds = np.mean(models_preds, axis = 0)

    # 1) AGBD prediction (Mg/ha) vs. GEDI reference (Mg/ha), with number of samples as point color
    fig = plt.figure(figsize = (10, 8))
    ma = 500
    h = plt.hist2d(labels.squeeze(), preds.squeeze(), bins = (100,100), cmap = 'Spectral_r', norm = LogNorm())
    plt.colorbar(h[3], label = 'Number of samples')
    plt.xlabel('GEDI reference AGBD [Mg/ha]')
    plt.ylabel('Prediction [Mg/ha]')
    plt.grid()
    plt.axis('equal')
    plt.plot([0,ma], [0,ma], 'k--')
    plt.xlim((0,ma))
    plt.ylim((0,ma))
    step = 25
    ticks = np.arange(0, ma + step, step)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.tight_layout()
    plt.savefig(join(plot_folder, f'{output_file_name}_AGBD.png'))
    plt.clf()

    # 3) Residuals vs GEDI reference, binned
    binned_residuals = []
    btw_bins = 50
    bins = np.arange(0, 501, btw_bins)
    for (lb, ub) in zip(bins[ : -1], bins[1 : ]):
        mask = (lb <= labels) & (labels <= ub)
        pred, label = preds[mask], labels[mask]
        if len(label) == 0 : binned_residuals.append(np.subtract(pred, label).tolist())
        else: binned_residuals.append(pred - label)
    plt.boxplot(binned_residuals, sym = '', positions = [x - btw_bins/2 for x in bins[1 : ]], widths = btw_bins/2)
    plt.grid()
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('GEDI Reference AGBD [Mg/ha]')
    plt.ylabel('Residuals ($pred_{AGBD}-GEDI_{AGBD}$) [Mg/ha]')
    plt.xticks(ticks = bins[:-1], labels = bins[:-1])
    plt.tight_layout()
    plt.savefig(join(plot_folder, f'{output_file_name}_Residuals.png'))
    plt.clf()