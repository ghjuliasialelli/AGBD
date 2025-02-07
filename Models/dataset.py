"""

This script defines the dataset class for the GEDI dataset.

"""

############################################################################################################################
# IMPORTS

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from os.path import join
import pickle
from os.path import join, exists
from datetime import datetime, timedelta
import argparse
np.seterr(divide = 'ignore') 
import tqdm
import pandas as pd

# Define the nodata values for each data source
NODATAVALS = {'S2_bands' : 0, 'CH': 255, 'ALOS_bands': 0, 'DEM': -9999, 'LC': 255}

# Define the biomes
REF_BIOMES = {20: 'Shrubs', 30: 'Herbaceous vegetation', 40: 'Cultivated', 90: 'Herbaceous wetland', 111: 'Closed-ENL', 112: 'Closed-EBL', 114: 'Closed-DBL', 115: 'Closed-mixed', 116: 'Closed-other', 121: 'Open-ENL', 122: 'Open-EBL', 124: 'Open-DBL', 125: 'Open-mixed', 126: 'Open-other'}

############################################################################################################################
# Helper functions

def initialize_index(fnames, mode, chunk_size, path_mapping, path_h5) :
    """
    This function creates the index for the dataset. The index is a dictionary which maps the file
    names (`fnames`) to the tiles that are in the `mode` (train, val, test); and the tiles to the
    number of chunks that make it up.

    Args:
    - fnames (list): list of file names
    - mode (str): the mode of the dataset (train, val, test)
    - chunk_size (int): the size of the chunks
    - path_mapping (str): the path to the file mapping each mode to its tiles

    Returns:
    - idx (dict): dictionary mapping the file names to the tiles and the tiles to the chunks
    - total_length (int): the total number of chunks in the dataset
    """

    # Load the mapping from mode to tile name
    with open(join(path_mapping, 'biomes_splits_to_name.pkl'), 'rb') as f:
        tile_mapping = pickle.load(f)

    # Iterate over all files
    idx = {}
    for fname in fnames :
        idx[fname] = {}
        
        with h5py.File(join(path_h5, fname), 'r') as f:
            
            # Get the tiles in this file which belong to the mode
            all_tiles = list(f.keys())
            tiles = np.intersect1d(all_tiles, tile_mapping[mode])
            
            # Iterate over the tiles
            for tile in tiles :

                # Get the number of patches in the tile
                n_patches = len(f[tile]['GEDI']['agbd'])
                idx[fname][tile] = n_patches // chunk_size
    
    total_length = sum(sum(v for v in d.values()) for d in idx.values())

    return idx, total_length


def find_index_for_chunk(index, n, total_length):
    """
    For a given `index` and `n`-th chunk, find the file, tile, and row index corresponding
    to this chunk.
    
    Args:
    - index (dict): dictionary mapping the files to the tiles and the tiles to the chunks
    - n (int): the n-th chunk

    Returns:
    - file_name (str): the name of the file
    - tile_name (str): the name of the tile
    - chunk_within_tile (int): the chunk index within the tile
    """

    # Check that the chunk index is within bounds
    assert n < total_length, "The chunk index is out of bounds"

    # Iterate over the index to find the file, tile, and row index
    cumulative_sum = 0
    for file_name, file_data in index.items():
        for tile_name, num_rows in file_data.items():
            if cumulative_sum + num_rows > n:
                # Calculate the row index within the tile
                chunk_within_tile = n - cumulative_sum
                return file_name, tile_name, chunk_within_tile
            cumulative_sum += num_rows


def encode_lat_lon(lat, lon) :
    """
    Encode the latitude and longitude into sin/cosine values. We use a simple WRAP positional encoding, as 
    Mac Aodha et al. (2019).

    Args:
    - lat (float): the latitude
    - lon (float): the longitude

    Returns:
    - (lat_cos, lat_sin, lon_cos, lon_sin) (tuple): the sin/cosine values for the latitude and longitude
    """

    # The latitude goes from -90 to 90
    lat_cos, lat_sin = np.cos(np.pi * lat / 90), np.sin(np.pi * lat / 90)
    # The longitude goes from -180 to 180
    lon_cos, lon_sin = np.cos(np.pi * lon / 180), np.sin(np.pi * lon / 180)

    # Now we put everything in the [0,1] range
    lat_cos, lat_sin = (lat_cos + 1) / 2, (lat_sin + 1) / 2
    lon_cos, lon_sin = (lon_cos + 1) / 2, (lon_sin + 1) / 2

    return lat_cos, lat_sin, lon_cos, lon_sin


def encode_coords(central_lat, central_lon, patch_size, resolution = 10) :
    """ 
    This function computes the latitude and longitude of a patch, from the latitude and longitude of its central pixel.
    It then encodes these values into sin/cosine values, and scales the results to [0,1].

    Args:
    - central_lat (float): the latitude of the central pixel
    - central_lon (float): the longitude of the central pixel
    - patch_size (tuple): the size of the patch
    - resolution (int): the resolution of the patch

    Returns:
    - (lat_cos, lat_sin, lon_cos, lon_sin) (tuple): the sin/cosine values for the latitude and longitude
    """

    # Initialize arrays to store latitude and longitude coordinates

    i_indices, j_indices = np.indices(patch_size)

    # Calculate the distance offset in meters for each pixel
    offset_lat = (i_indices - patch_size[0] // 2) * resolution
    offset_lon = (j_indices - patch_size[1] // 2) * resolution

    # Calculate the latitude and longitude for each pixel
    # cf. https://stackoverflow.com/questions/7477003/calculating-new-longitude-latitude-from-old-n-meters 
    # the volumetric mean radius of the Earth is 6371km, cf. https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html 
    latitudes = central_lat + (offset_lat / 6371000) * (180 / np.pi)
    longitudes = central_lon + (offset_lon / 6371000) * (180 / np.pi) / np.cos(latitudes * np.pi / 180)

    # Encode the latitude and longitude
    lat_cos, lat_sin, lon_cos, lon_sin = encode_lat_lon(latitudes, longitudes)

    return lat_cos, lat_sin, lon_cos, lon_sin


def get_doy(num_days, patch_size, GEDI_START_MISSION = '2019-04-17') :
    """
    For a given number of days before/since the start of the GEDI mission, this function calculates
    the day of year (number between 1 and 365) and encodes it into sin/cosine values.

    Args:
    - num_days (int): the number of days before/since the start of the GEDI mission
    - GEDI_START_MISSION (str): the start date of the GEDI mission

    Returns:
    - (doy_cos, doy_sin) (tuple): the sin/cosine values for the day of year (doy_cos, doy_sin
    """

    # Get the date of acquisition and day of year
    start_date = datetime.strptime(GEDI_START_MISSION, '%Y-%m-%d')
    target_date = start_date + timedelta(days = int(num_days))
    doy = target_date.timetuple().tm_yday - 1 # range [1, 366]

    # Get the doy_cos and doy_sin
    doy_cos = np.cos(2 * np.pi * doy / 365)
    doy_sin = np.sin(2 * np.pi * doy / 365)

    # Now we put everything in the [0,1] range
    doy_cos, doy_sin = (doy_cos + 1) / 2, (doy_sin + 1) / 2

    return np.full((patch_size[0], patch_size[1]), doy_cos), np.full((patch_size[0], patch_size[1]), doy_sin)


def func_slope(px, py) :
    return np.sqrt(px ** 2 + py ** 2)

def func_aspect(px, py) :
    aspect = np.pi / 2 - np.arctan2(py, px)
    return np.where(aspect < 0, aspect + 2 * np.pi, aspect)

def get_topology(dem) :
    """
    This function computes the slope and aspect of the DEM.
    
    Resources: 
    . https://www.spatialanalysisonline.com/HTML/gradient__slope_and_aspect.htm
    . https://gis.stackexchange.com/questions/361837/calculating-slope-of-numpy-array-using-gdal-demprocessing
    . https://math.stackexchange.com/a/3923660

    Args:
    - dem (np.array, shape batch_size, patch_size, patch_size): the DEM

    Returns:
    - slope (np.array): the slope of the DEM
    - aspect_cos (np.array): the cosine of the aspect of the DEM
    - aspect_sin (np.array): the sine of the aspect of the DEM
    """

    # Get the partial derivatives
    px, py = np.gradient(dem, 10,)
    # Get the slope, in [0,1]
    slope = np.sqrt(px ** 2 + py ** 2)
    # Get the aspect, in [0,2pi]
    aspect = np.pi / 2 - np.arctan2(py, px)
    aspect = np.where(aspect < 0, aspect + 2 * np.pi, aspect)
    # Encode and scale the aspect, in [0,1]
    aspect_cos = (np.cos(aspect) + 1) / 2
    aspect_sin = (np.sin(aspect) + 1) / 2
    
    return slope, aspect_cos, aspect_sin


def normalize_data(data, norm_values, norm_strat, nodata_value = None) :
    """
    Normalize the data, according to various strategies:
    - mean_std: subtract the mean and divide by the standard deviation
    - pct: subtract the 1st percentile and divide by the 99th percentile
    - min_max: subtract the minimum and divide by the maximum

    Args:
    - data (np.array): the data to normalize
    - norm_values (dict): the normalization values
    - norm_strat (str): the normalization strategy

    Returns:
    - normalized_data (np.array): the normalized data
    """

    if norm_strat == 'mean_std' :
        mean, std = norm_values['mean'], norm_values['std']
        if nodata_value is not None :
            data = np.where(data == nodata_value, 0, (data - mean) / std)
        else : data = (data - mean) / std

    elif norm_strat == 'pct' :
        p1, p99 = norm_values['p1'], norm_values['p99']
        if nodata_value is not None :
            data = np.where(data == nodata_value, 0, (data - p1) / (p99 - p1))
        else :
            data = (data - p1) / (p99 - p1)
        data = np.clip(data, 0, 1)

    elif norm_strat == 'min_max' :
        min_val, max_val = norm_values['min'], norm_values['max']
        if nodata_value is not None :
            data = np.where(data == nodata_value, 0, (data - min_val) / (max_val - min_val))
        else:
            data = (data - min_val) / (max_val - min_val)
    
    else: 
        raise ValueError(f'Normalization strategy `{norm_strat}` is not valid.')

    return data


def normalize_bands(bands_data, norm_values, order, norm_strat, nodata_value = None) :
    """
    This function normalizes the bands data using the normalization values and strategy.

    Args:
    - bands_data (np.array): the bands data to normalize
    - norm_values (dict): the normalization values
    - order (list): the order of the bands
    - norm_strat (str): the normalization strategy
    - nodata_value (int/float): the nodata value

    Returns:
    - bands_data (np.array): the normalized bands data
    """
    
    for i, band in enumerate(order) :
        band_norm = norm_values[band]
        bands_data[:, :, i] = normalize_data(bands_data[:, :, i], band_norm, norm_strat, nodata_value)
    
    return bands_data


def encode_lc(lc_data) :
    """
    Encode the land cover classes into sin/cosine values and scale the class probabilities to [0,1].

    Args:
    - lc_data (np.array): the land cover data

    Returns:
    - lc_cos (np.array): the cosine values of the land cover classes
    - lc_sin (np.array): the sine values of the land cover classes
    - lc_prob (np.array): the land cover class probabilities
    """

    # Get the land cover classes
    lc_map = lc_data[:, :, 0]

    # Encode the LC classes with sin/cosine values and scale the data to [0,1]
    lc_cos = np.where(lc_map == NODATAVALS['LC'], 0, (np.cos(2 * np.pi * lc_map / 100) + 1) / 2)
    lc_sin = np.where(lc_map == NODATAVALS['LC'], 0, (np.sin(2 * np.pi * lc_map / 100) + 1) / 2)

    # Scale the class probabilities to [0,1]
    lc_prob = lc_data[:, :, 1]
    lc_prob = np.where(lc_prob == NODATAVALS['LC'], 0, lc_prob / 100)

    return lc_cos, lc_sin, lc_prob


def embed_lc(lc_data, embeddings) :
    """
    Embed the land cover classes using the cat2vec embeddings.

    Args:
    - lc_data (np.array): the land cover data
    - embeddings (dict): the cat2vec embeddings

    Returns:
    - lc_map (np.array): the embedded land cover classes
    - lc_prob (np.array): the land cover class probabilities
    """

    # Get the land cover classes
    lc_map = lc_data[:, :, 0]
    lc_map = np.vectorize(lambda x: embeddings.get(x, embeddings.get(0)), signature = '()->(n)')(lc_map)

    # Scale the class probabilities to [0,1]
    lc_prob = lc_data[:, :, 1]
    lc_prob = np.where(lc_prob == NODATAVALS['LC'], 0, lc_prob / 100)

    return lc_map, lc_prob


_biome_values_mapping = {v: i for i, v in enumerate(REF_BIOMES.keys())}
def onehot_lc(lc_data) :
    """
    Encode the land cover classes using one-hot encoding.

    Args:
    - lc_data (np.array): the land cover data

    Returns:
    - lc_map (np.array): the one-hot encoded land cover classes
    """
    # Number of classes
    num_classes = len(_biome_values_mapping)
    # Actually perform the one-hot encoding
    def one_hot(x) :
        one_hot = np.zeros(num_classes)
        one_hot[_biome_values_mapping.get(x, 0)] = 1
        return one_hot
    one_hot_data = np.vectorize(one_hot, signature = '() -> (n)')(lc_data).astype(np.float32)
    return one_hot_data


_ref_biome_values = [v for v in REF_BIOMES.keys()]
def biome_distribution(patch_lc) :
    """
    This function computes the distribution of biomes in a patch.

    Args:
    - patch_lc (np.array): the land cover classes in the patch, of size (patch_size, patch_size)

    Returns:
    - biome_emb (np.array): the biome distribution, of size (num_classes,)
    """
    # Number of pixels in the patch
    num_pixels = patch_lc.size
    # Percentage of each biome in the patch
    counts = {value: np.count_nonzero(patch_lc == value) / num_pixels for value in _ref_biome_values}
    return np.array(list(counts.values())).astype(np.float32)


class GEDIDataset(Dataset):

    def __init__(self, paths, years, chunk_size, mode, args, version = 4, debug = False):

        # Get the parameters
        self.h5_path, self.norm_path, self.mapping, self.embed_path = paths['h5'], paths['norm'], paths['map'], paths['embeddings']
        self.mode = mode
        self.chunk_size = chunk_size
        self.years = years
        
        # Get the file names
        self.fnames = []
        for year in self.years : 
            if debug : self.fnames += [f'data_subset-{year}-v{version}_{i}-20.h5' for i in range(2)]
            else: self.fnames += [f'data_subset-{year}-v{version}_{i}-20.h5' for i in range(20)]
        
        # Initialize the index
        self.index, self.length = initialize_index(self.fnames, self.mode, self.chunk_size, self.mapping, self.h5_path)

        # Define the data to use
        self.latlon = args.latlon
        self.bands = args.bands
        self.ch = args.ch
        self.s1 = args.s1
        self.alos = args.alos
        self.lc = args.lc
        self.cat2vec = args.cat2vec
        self.onehot = args.onehot
        self.dist = args.dist
        self.dem = args.dem
        self.gedi_dates = args.gedi_dates
        self.s2_dates = args.s2_dates
        self.s2_day = args.s2_day
        self.s2_doy = args.s2_doy
        self.topo = args.topo
        self.aspect = args.aspect
        self.slope = args.slope
        self.patch_size = args.patch_size

        # Define the learning procedure
        self.norm_strat = args.norm_strat
        self.norm_target = args.norm

        # Check that the mode is valid
        assert self.mode in ['train', 'val', 'test'], "The mode must be one of 'train', 'val', 'test'"

        # Load the normalization values
        if not exists(join(self.norm_path, f"statistics_subset_2019-2020-v{version}_new.pkl")):
            raise FileNotFoundError(f"The file `statistics_subset_2019-2020-v{version}_new.pkl` does not exist.")
        with open(join(self.norm_path, f"statistics_subset_2019-2020-v{version}_new.pkl"), mode = 'rb') as f:
            self.norm_values = pickle.load(f)

        # Open the file handles
        self.handles = {fname: h5py.File(join(self.h5_path, fname), 'r') for fname in self.index.keys()}

        # Define the window size
        assert self.patch_size[0] == self.patch_size[1], "The patch size must be square"
        self.center = 12 # because the patch size is 25x25 in the .h5 files
        self.window_size = self.patch_size[0] // 2

        # Get the cat2vec LC embeddings
        if self.lc and self.cat2vec :
            embeddings = pd.read_csv(join(self.embed_path, "embeddings_train.csv"))
            embeddings = dict([(v,np.array([a,b,c,d,e])) for v, a,b,c,d,e in zip(embeddings.mapping, embeddings.dim0, embeddings.dim1, embeddings.dim2, embeddings.dim3, embeddings.dim4)])
            self.embeddings = embeddings

    def __len__(self):
        return self.length
    
    def __getitem__(self, n):
            
        # Find the file, tile, and row index corresponding to this chunk
        file_name, tile_name, idx = find_index_for_chunk(self.index, n, self.length)
        
        # Get the file handle
        f = self.handles[file_name]

        # Set the order and indices for the Sentinel-2 bands
        if not hasattr(self, 's2_order') : self.s2_order = list(f[tile_name]['S2_bands'].attrs['order'])
        if not hasattr(self, 's2_indices') : self.s2_indices = [self.s2_order.index(band) for band in self.bands]

        # Set the order for the Sentinel-1 bands
        if self.s1 and not hasattr(self, 's1_order') : self.s1_order = f[tile_name]['S1_bands'].attrs['order']

        # Set the order for the ALOS bands
        if self.alos and not hasattr(self, 'alos_order') : self.alos_order = f[tile_name]['ALOS_bands'].attrs['order']

        # Initialize the data list
        data = []

        # Sentinel-2 bands
        if self.bands != [] :
            
            # Get the bands
            s2_bands = f[tile_name]['S2_bands'][idx, self.center - self.window_size : self.center + self.window_size + 1, self.center - self.window_size : self.center + self.window_size + 1, :].astype(np.float32)
            
            # Get the BOA offset, if it exists
            if 'S2_boa_offset' in f[tile_name]['Sentinel_metadata'].keys() : 
                s2_boa_offset = f[tile_name]['Sentinel_metadata']['S2_boa_offset'][idx]
            else: s2_boa_offset = 0

            # Get the surface reflectance values
            sr_bands = (s2_bands - s2_boa_offset * 1000) / 10000
            sr_bands[s2_bands == 0] = 0
            sr_bands[sr_bands < 0] = 0
            s2_bands = sr_bands

            # Normalize the bands
            s2_bands = normalize_bands(s2_bands, self.norm_values['S2_bands'], self.s2_order, self.norm_strat, NODATAVALS['S2_bands'])
            s2_bands = s2_bands[:, :, self.s2_indices]
            data.extend([s2_bands])
            
            # Sentinel-2 date
            s2_num_days = f[tile_name]['Sentinel_metadata']['S2_date'][idx]
            if self.s2_dates : 
                s2_doy_cos, s2_doy_sin = get_doy(s2_num_days, self.patch_size)
                s2_num_days = np.full((self.patch_size[0], self.patch_size[1]), s2_num_days).astype(np.float32)
                s2_num_days = normalize_data(s2_num_days, self.norm_values['Sentinel_metadata']['S2_date'], 'min_max' if self.norm_strat == 'pct' else self.norm_strat)
                if self.s2_day:
                    data.extend([s2_num_days[..., np.newaxis]])
                if self.s2_doy:
                    data.extend([s2_doy_cos[..., np.newaxis], s2_doy_sin[..., np.newaxis]])
            
        # Sentinel-1 bands
        if self.s1:
            s1_bands = f[tile_name]['S1_bands'][idx, self.center - self.window_size : self.center + self.window_size + 1, self.center - self.window_size : self.center + self.window_size + 1, :].astype(np.float32)
            s1_bands = normalize_bands(s1_bands, self.norm_values['S1_bands'], self.s1_order, self.norm_strat)
            
            s1_num_days = f[tile_name]['Sentinel_metadata']['S1_date'][idx, :]
            s1_doy_cos, s1_doy_sin = get_doy(s1_num_days, self.patch_size)
            s1_num_days = np.full((self.patch_size[0], self.patch_size[1]), s1_num_days).astype(np.float32)
            s1_num_days = normalize_data(s1_num_days, self.norm_values['Sentinel_metadata']['S1_date'], 'min_max' if self.norm_strat == 'pct' else self.norm_strat)
            
            data.extend([s1_bands, s1_num_days[..., np.newaxis], s1_doy_cos[..., np.newaxis], s1_doy_sin[..., np.newaxis]])
        
        # Latitude and longitude data
        lat_offset, lat_decimal = f[tile_name]['GEDI']['lat_offset'][idx], f[tile_name]['GEDI']['lat_decimal'][idx]
        lon_offset, lon_decimal = f[tile_name]['GEDI']['lon_offset'][idx], f[tile_name]['GEDI']['lon_decimal'][idx]
        lat = np.sign(lat_decimal) * (np.abs(lat_decimal) + lat_offset)
        lon = np.sign(lon_decimal) * (np.abs(lon_decimal) + lon_offset)
        lat_cos, lat_sin, lon_cos, lon_sin = encode_coords(lat, lon, self.patch_size)
        if self.latlon : data.extend([lat_cos[..., np.newaxis], lat_sin[..., np.newaxis], lon_cos[..., np.newaxis], lon_sin[..., np.newaxis]])
        else: data.extend([lat_cos[..., np.newaxis], lat_sin[..., np.newaxis]])
        
        # GEDI dates
        gedi_num_days = f[tile_name]['GEDI']['date'][idx]

        if self.gedi_dates :
            gedi_doy_cos, gedi_doy_sin = get_doy(gedi_num_days, self.patch_size)
            gedi_num_days = np.full((self.patch_size[0], self.patch_size[1]), gedi_num_days).astype(np.float32)
            gedi_num_days = normalize_data(gedi_num_days, self.norm_values['GEDI']['date'], 'min_max' if self.norm_strat == 'pct' else self.norm_strat)
            data.extend([gedi_num_days[..., np.newaxis], gedi_doy_cos[..., np.newaxis], gedi_doy_sin[..., np.newaxis]])

        # ALOS bands
        if self.alos:

            # Get the bands
            alos_bands = f[tile_name]['ALOS_bands'][idx, self.center - self.window_size : self.center + self.window_size + 1, self.center - self.window_size : self.center + self.window_size + 1, :].astype(np.float32)

            # Get the gamma naught values
            alos_bands = np.where(alos_bands == NODATAVALS['ALOS_bands'], -9999.0, 10 * np.log10(np.power(alos_bands.astype(np.float32), 2)) - 83.0)

            # Normalize the bands
            alos_bands = normalize_bands(alos_bands, self.norm_values['ALOS_bands'], self.alos_order, self.norm_strat, -9999.0)

            data.extend([alos_bands])
        
        # CH data
        if self.ch:
            ch = f[tile_name]['CH']['ch'][idx, self.center - self.window_size : self.center + self.window_size + 1, self.center - self.window_size : self.center + self.window_size + 1]
            ch = normalize_data(ch, self.norm_values['CH']['ch'], self.norm_strat, NODATAVALS['CH'])
            
            ch_std = f[tile_name]['CH']['std'][idx, self.center - self.window_size : self.center + self.window_size + 1, self.center - self.window_size : self.center + self.window_size + 1]
            ch_std = normalize_data(ch_std, self.norm_values['CH']['std'], self.norm_strat, NODATAVALS['CH'])

            data.extend([ch[..., np.newaxis], ch_std[..., np.newaxis]])
        
        # LC data
        if self.lc:
            lc = f[tile_name]['LC'][idx, self.center - self.window_size : self.center + self.window_size + 1, self.center - self.window_size : self.center + self.window_size + 1, :]
            
            if self.cat2vec: # cat2vec embeddings, 5dim
                lc, lc_prob = embed_lc(lc, self.embeddings)
                data.extend([lc, lc_prob[..., np.newaxis]])
            
            elif self.onehot: # one-hot encoding, 14dim
                
                # Get the probability layer
                lc_prob = np.where(lc[:, :, 1] == NODATAVALS['LC'], 0, lc[:, :, 1] / 100)

                # Get the one-hot encoding of the biome layer
                if self.dist: # with pct
                    lc = biome_distribution(lc[:, :, 0])
                else:
                    lc = onehot_lc(lc[:, :, 0])

                data.extend([lc, lc_prob[..., np.newaxis]])
            
            else: # sin/cosine encoding, 2dim
                lc_cos, lc_sin, lc_prob = encode_lc(lc)
                data.extend([lc_cos[..., np.newaxis], lc_sin[..., np.newaxis], lc_prob[..., np.newaxis]])
        
        # DEM data
        if self.dem:
            dem = f[tile_name]['DEM'][idx, self.center - self.window_size : self.center + self.window_size + 1, self.center - self.window_size : self.center + self.window_size + 1]

            if self.topo :
                # Get the slope and aspect
                slope, aspect_cos, aspect_sin = get_topology(dem)
                if self.slope :
                    data.extend([slope[..., np.newaxis]])
                if self.aspect:
                    data.extend([aspect_cos[..., np.newaxis], aspect_sin[..., np.newaxis]])

            dem = normalize_data(dem, self.norm_values['DEM'], self.norm_strat, NODATAVALS['DEM'])
            data.extend([dem[..., np.newaxis]])
        
        # Concatenate the data together
        data = torch.from_numpy(np.concatenate(data, axis = -1).swapaxes(-1, 0)).to(torch.float)

        # Get the GEDI target data
        agbd = f[tile_name]['GEDI']['agbd'][idx]
        if self.norm_target :
            agbd = normalize_data(agbd, self.norm_values['GEDI']['agbd'], self.norm_strat)
        agbd = torch.from_numpy(np.array(agbd, dtype = np.float32)).to(torch.float)

        return data, agbd