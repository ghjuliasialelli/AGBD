"""

This script performs inference on a Sentinel-2 tile, using a trained model. The model is loaded from the checkpoint
folder, and the input data is loaded from the patches folder. The predictions are saved in the saving_dir folder.

Launch the script with `bash inference.sh`.

"""

#######################################################################################################################
# Imports

import time
from os.path import join
import os, pickle, argparse
import torch
import numpy as np
import rasterio as rs
from skimage.transform import rescale
from torch import set_float32_matmul_precision
from models import Net
from wrapper import Model
from torch import set_float32_matmul_precision
from inference_helper import *
from dataset import normalize_bands, normalize_data, encode_lc
import warnings
from os import getcwd

# Silencing specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")


#######################################################################################################################
# Helper functions 

def inf_parser():
    """ 
    Parse the command line arguments for the inference script.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type = str, required = True, help = 'Path to the dataset')
    parser.add_argument('--model', type = str, required = True, help = 'Model names')
    parser.add_argument('--arch', type = str, required = True, help = 'Architecture of the model')
    parser.add_argument('--saving_dir', type = str, help = 'Directory in which to save the plots.')
    parser.add_argument("--tile_name", required = True, type = str, help = 'Tile on which to run the prediction.')
    parser.add_argument("--dw", action = 'store_true', help = 'Downsample the preds to 50m resolution.')
    parser.add_argument("--patch_size", nargs = 2, type = int, default = [200,200], help = 'Size (height,width) of the patches.')
    parser.add_argument("--overlap_size", nargs = 2, type = int, default = [100,100], help = 'Size (height,width) of the patches.')
    args = parser.parse_args()

    return args, args.dataset_path, args.model, args.arch, args.saving_dir, args.tile_name, args.dw, args.patch_size, args.overlap_size


def load_input(paths, tile_name, norm_values, cfg, alos_order = ['HH', 'HV']):
    """ 
    Load the input data for the inference. 

    Args:
    - paths (dict) : dictionary with keys `norm`, `tiles`, and `ckpt` and with values
        the paths to the corresponding file/folder
    - tile_name (str) : the name of the Sentinel-2 tile to perform inference on
    - norm_values (dict) : dictionary with the normalization values for the different bands
    - cfg (dict) : dictionary with the configuration of the model
    - alos_order (list) : list with the order of the ALOS bands

    Returns:
    - data (torch.Tensor) : the input data for the model
    - mask (np.array) : the mask of the input data
    - meta (dict) : the metadata of the input data
    """
    
    start_time = time.time()
    print('Loading input...')


    data = []

    # Sentinel 2 bands -------------------------------------------------------------------------------------------
    # 1. Get the product
    with open(join(paths['tiles'], 'mapping.pkl'), 'rb') as f: least_cloudy_products = pickle.load(f)
    s2_prod = least_cloudy_products[tile_name]
    year = s2_prod.split('_')[2][:4]
    # 2. Process the product
    transform, upsampling_shape, s2_bands, crs, bounds, boa_offset, lat_cos, lat_sin, lon_cos, lon_sin, meta = process_S2_tile(s2_prod, paths['tiles'])
    scl_band = s2_bands.pop('SCL')
    # 3. Get the SR values for the optical bands
    for band, band_value in s2_bands.items() :
        s2_bands[band] = (band_value - boa_offset * 1000) / 10000
    # 4. Normalize the data
    s2_order = cfg['bands']
    s2_bands = np.moveaxis(np.array([s2_bands[band] for band in s2_order]), 0, -1)
    s2_bands = normalize_bands(s2_bands, norm_values['S2_bands'], s2_order, cfg['norm_strat'], NODATAVALS['S2'])
    
    # Get the ALOS data ------------------------------------------------------------------------------------------
    # 1. Get the data
    alos_raw = load_ALOS_data(tile_name, paths['alos'], year)
    alos_tile = get_tile(alos_raw, transform, upsampling_shape, 'ALOS', ALOS_attrs)
    alos_bands = np.moveaxis(np.array([alos_tile['HH'], alos_tile['HV']]), 0, -1)
    # 2. Get the gamma naught values
    alos_bands = np.where(alos_bands == NODATAVALS['ALOS'], -9999.0, 10 * np.log10(np.power(alos_bands.astype(np.float32), 2)) - 83.0)
    # 3. Normalize the data
    alos_bands = normalize_bands(alos_bands, norm_values['ALOS_bands'], alos_order, cfg['norm_strat'], -9999.0)

    # Get the CH data and latitude / longitude -------------------------------------------------------------------
    # 1. Get the data
    ch_bands = load_CH_data(paths['ch'], tile_name, year)
    ch, ch_std = ch_bands['ch'], ch_bands['std']
    # 2. Normalize the data
    ch = normalize_data(ch, norm_values['CH']['ch'], cfg['norm_strat'], NODATAVALS['CH'])
    ch_std = normalize_data(ch_std, norm_values['CH']['std'], cfg['norm_strat'], NODATAVALS['CH'])
    
    # Get the LC data --------------------------------------------------------------------------------------------
    # 1. Get the data
    lc_raw = load_LC_data(paths['lc'], tile_name)
    lc_tile = get_tile(lc_raw, transform, upsampling_shape, 'LC', LC_attrs)
    lc = np.moveaxis(np.array([lc_tile['lc'], lc_tile['prob']]), 0, -1)
    # 2. Transform the data
    lc_cos, lc_sin, lc_prob = encode_lc(lc)

    # Get the DEM data -------------------------------------------------------------------------------------------
    # 1. Get the data
    dem_raw = load_DEM_data(paths['dem'], tile_name)
    dem_tile = get_tile(dem_raw, transform, upsampling_shape, 'DEM', DEM_attrs)
    dem = dem_tile['dem']
    # 2. Normalize the data
    dem = normalize_data(dem, norm_values['DEM'], cfg['norm_strat'], NODATAVALS['DEM'])

    # Add all of the data in the correct order -------------------------------------------------------------------
    if cfg['bands'] != [] : data.extend([s2_bands])
    if cfg['latlon']: data.extend([lat_cos[..., np.newaxis], lat_sin[..., np.newaxis], lon_cos[..., np.newaxis], lon_sin[..., np.newaxis]])
    else: data.extend([lat_cos[..., np.newaxis], lat_sin[..., np.newaxis]])
    if cfg['alos'] : data.extend([alos_bands])
    if cfg['ch'] : data.extend([ch[..., np.newaxis], ch_std[..., np.newaxis]])
    if cfg['lc'] : data.extend([lc_cos[..., np.newaxis], lc_sin[..., np.newaxis], lc_prob[..., np.newaxis]])
    if cfg['dem'] : data.extend([dem[..., np.newaxis]])
    data = torch.from_numpy(np.concatenate(data, axis = -1)).to(torch.float)
    
    # Get the mask ------------------------------------------------------------------------------------------------
    # i.e. where it is Water (6) and Snow or ice (11)
    mask = (scl_band == 6) | (scl_band == 11)

    print('done!')
    end_time = time.time()
    print(f'Loading input took {end_time - start_time} seconds.')

    return data, mask, meta


def predict_patch(model, patch, device):
    """
    Predict the AGBD of a patch using the model.

    Args:
    - model (torch.nn.Module) : the model to use for the prediction
    - patch (torch.Tensor) : the patch to predict on
    - device (torch.device) : the device to use for the prediction

    Returns:
    - preds (np.array) : the predictions of the model on the patch
    """

    patch = torch.unsqueeze(torch.permute(patch, [2,0,1]), 0).to(device)
    preds = model.model(patch).cpu().detach().numpy()
    return preds[0, 0, :, :]


def predict_tile(img, size, model, patch_size, overlap_size, device):
    """
    Predict the AGBD of a Sentinel-2 tile using the model. We split the ~ 100km x 100km tile into patches of size
    `patch_size`, with overlap by `overlap_size`. Best practices: . choose patch_size such that patch_size / 5 is
    an integer; . choose overlap_size such that overlap_size / 2 is an integer.

    Args:
    - img (np.array) : the Sentinel-2 tile to predict on
    - size (int) : the size of the Sentinel-2 tile
    - model (torch.nn.Module) : the model to use for the prediction
    - patch_size (tuple) : the size of the patches to use for the prediction
    - overlap_size (tuple) : the size of the overlap between the patches
    - device (torch.device) : the device to use for the prediction

    Returns:
    - predictions (np.array) : the predictions of the model on the Sentinel-2 tile
    """

    # Define variables for the splitting of the Sentinel-2 tile into patches ######################################
    
    # Width and height of the input Sentinel-2 tile
    img_height, img_width, _ = img.shape
    # Width and height of the desired patches
    patch_height, patch_width = patch_size
    # Width and height of the desired overlap between two patches
    overlap_height, overlap_width = overlap_size
    # Step in the width/height dimension: width/height of the patch minus width/height of the overlap
    step_height, step_width = patch_height - overlap_height, patch_width - overlap_width
    # Find the number of times the patch will fit entirely in the image
    n_height, n_width = (img_height - overlap_height) / (patch_height - overlap_height), (img_width - overlap_width) / (patch_width - overlap_width)
    overload_height, overload_width = True, True
    if (n_height % 1) == 0 : overload_height = False
    else: n_height = np.ceil(n_height)
    if (n_width % 1) == 0 : overload_width = False
    else: n_width = np.ceil(n_width)

    # Define variables for the predictions mosaicing ##############################################################
    
    # Downsampling factor, to predict at a 50m resolution per pixel if enabled
    dw_factor = 15 // size
    # Width and height of the prediction patch: the (downsampled) width/height of the patch
    pred_patch_width, pred_patch_height  = int(np.ceil(patch_width  / dw_factor)), int(np.ceil(patch_height / dw_factor))
    # Width and height of the prediction patch overlap: the (downsampled) width/height of the overlap
    pred_overlap_width, pred_overlap_height  = int(np.ceil(overlap_width  / dw_factor)), int(np.ceil(overlap_height / dw_factor))
    # Width and height of the mosaiced predictions: the (downsampled) width/height of the Sentinel-2 tile
    pred_width, pred_height = img_width // dw_factor, img_height // dw_factor
    # Step in the width/height dimension: width/height of the prediction patch minus width/height of the prediction overlap
    pred_step_width, pred_step_height = pred_patch_width - pred_overlap_width, pred_patch_height - pred_overlap_height
    
    
    # Place-holder for the put-together predictions
    predictions = np.zeros((pred_height, pred_width), dtype = np.float32)

    # Actual prediction ###########################################################################################

    print('Actual tile prediction...')
    start_time = time.time()
    # Iterate over the patches and predict the AGBD
    for i, i_p in zip(range(0, img_height - patch_height + 1, step_height), range(0, pred_height - pred_patch_height + 1, pred_step_height)) :
        off_h = 0 if i_p == 0 else overlap_height // (2 * dw_factor) # to limit border-effect
        for j, j_p in zip(range(0, img_width - patch_width + 1, step_width), range(0, pred_width - pred_patch_width + 1, pred_step_width)) :
            off_w = 0 if j_p == 0 else overlap_width // (2 * dw_factor) # to limit border-effect
            patch = img[i : i + patch_height, j : j + patch_width, :]
            predictions[i_p + off_h : i_p + pred_patch_height, j_p + off_w : j_p + pred_patch_width] = predict_patch(model, patch, device)[off_h : , off_w :]
        # Last column, if patches don't equally fit in the image
        if overload_width :
            patch = img[i : i + patch_height, - patch_width : , :]
            predictions[i_p + off_h : i_p + pred_patch_height, - pred_patch_width + off_w : ] = predict_patch(model, patch, device)[off_h : , off_w :]
    # Last row, if patches don't equally fit in the image
    if overload_height :
        for j, j_p in zip(range(0, img_width - patch_width + 1, step_width), range(0, pred_width - pred_patch_width + 1, pred_step_width)) :
            patch = img[ - patch_height : , j : j + patch_width, :]
            predictions[- pred_patch_height + off_h : , j_p + off_w : j_p + pred_patch_width] = predict_patch(model, patch, device)[off_h : , off_w :]
    
    # Divide by the number of times a value was in an overlap to get the mean
    print('done!')
    end_time = time.time()
    print(f'Actual tile prediction took {end_time - start_time} seconds.')
    return predictions


#######################################################################################################################
# Inference class definition

class Inference:

    """ 
    An `Inference` object loads a PyTorch model and performs AGBD inference at the Sentinel-2 tile level.
    """

    def __init__(self, arch, model_name, paths, tile_name, args, device):

        """
        Initialization method.

        Args:
        - arch (str) : the architecture of the model
        - model_name (str) : the name of the model
        - paths (dict) : dictionary with the paths to the different folders
        - tile_name (str) : the name of the Sentinel-2 tile to perform inference on
        - args (argparse.Namespace) : the arguments of the model
        - device (torch.device) : the device to use for the inference
        """

        self.arch = arch
        self.model_name = model_name
        self.paths = paths
        self.tile_name = tile_name
        self.args = args     
        self.device = device   
        self.load_model()
    
    def load_model(self):  # sourcery skip: raise-specific-error

        """ 
        Loads the model, setting self.model
        """

        # Initialize the model
        model = Net(model_name = self.arch, in_features = self.args.in_features, num_outputs = self.args.num_outputs, 
                    channel_dims = self.args.channel_dims, max_pool = self.args.max_pool, downsample = None,
                    leaky_relu = self.args.leaky_relu, patch_size = self.args.patch_size)
        
        model = Model(model, lr = self.args.lr, step_size = self.args.step_size, gamma = self.args.gamma, 
                        patch_size = self.args.patch_size, downsample = self.args.downsample, 
                        loss_fn = self.args.loss_fn)
    
        state_dict = torch.load(join(self.paths['ckpt'], self.arch, f'{self.model_name}_best.ckpt'), map_location = torch.device('cpu'))['state_dict']
        try: model.load_state_dict(state_dict) 
        except Exception as e:
            if self.arch == 'nico' :
                state_dict = {k.replace('_orig_mod.',''):v for k,v in state_dict.items()}
                model.load_state_dict(state_dict)
            else: raise e
        model.to(self.device)
        model.eval()
        model.model.eval()
        self.model = model.model

#######################################################################################################################
# Code execution

def run_inference():
    """
    This function runs the inference on a Sentinel-2 tile, using a trained model. The model is loaded from the checkpoint
    folder, and the input data is loaded from the patches folder. The predictions are saved in the saving_dir folder.
    """
    
    # Get the command line arguments and set the global variables
    args, dataset_path, model, arch, saving_dir, tile_name, dw, patch_size, overlap_size = inf_parser()

    # Settings
    set_float32_matmul_precision('high')
    if (dataset_path == 'local') : accelerator, cpus_per_task = 'auto', 8
    else: accelerator, cpus_per_task = 'gpu', int(os.environ.get('SLURM_CPUS_PER_TASK'))
    if cpus_per_task is None: cpus_per_task = 16
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the paths
    if dataset_path == 'local' :
        inference_dir = join(getcwd(), 'inference')
        dataset_path = {'norm': inference_dir,
                           'tiles': inference_dir,
                           'ch': inference_dir,
                           'ckpt': join(getcwd(), 'pretrained_weights'),
                           'alos': inference_dir,
                           'dem': inference_dir,
                           'lc': inference_dir,
                           }
    else: 
        raise NotImplementedError('Specify your own paths.')
    dataset_path['saving_dir'] = saving_dir

    # We get the config for one of the models
    with open(join(dataset_path['ckpt'], arch, f'{model}_cfg.pkl'), 'rb') as f:
        cfg = pickle.load(f)
    for key, value in cfg.items(): setattr(args, key, value)

    # Load the input
    with open(os.path.join(dataset_path['norm'], 'statistics_subset_2019-2020-v4.pkl'), mode = 'rb') as f: norm_values = pickle.load(f)
    img, mask, meta = load_input(dataset_path, tile_name, norm_values, cfg)
    size = 3 if cfg['downsample'] else 15
    pred_mask = rescale(mask, size / 15)

    # Load the models
    inference_object = Inference(arch = arch, model_name = model, paths = dataset_path, tile_name = tile_name, args = args, device = device)
    inf_model = inference_object.model

    # Get the predictions
    predictions = predict_tile(img, size, inf_model, patch_size, overlap_size, device)
    predictions[pred_mask] = np.nan

    # Cast negative AGB values to 0, and all values to uint16
    predictions[predictions < 0] = 0
    predictions[predictions > 65535] = 65535
    predictions[np.isinf(predictions)] = 65535
    predictions[np.isnan(predictions)] = 65535
    predictions = predictions.astype(np.uint16)

    # Save the AGB predictions to a GeoTIFF, with dtype uint16
    meta.update(driver = 'GTiff', dtype = np.uint16, count = 1, compress = 'lzw', nodata = 65535)
    print(f'Saving predictions to {os.path.join(dataset_path["saving_dir"], f"{arch}_{tile_name}_<agb/std>.tif")}')
    with rs.open(os.path.join(dataset_path["saving_dir"], f'{arch}_{tile_name}_agb.tif'), 'w', **meta) as f:
        f.write(predictions, 1)

if __name__ == '__main__': 
    run_inference()
    print('Inference done!')