"""

This script is used to extract the features and labels of the central pixels from the h5 files and save them as csv files.

"""

############################################################################################################################
# IMPORTS

from dataset import *
import argparse
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

############################################################################################################################
# Helper functions

def construct_feature_names(args) :
    """
    This function constructs the feature names based on the arguments.

    Args:
    - args : argparse.Namespace : the arguments

    Returns:
    - feature_names : list : the feature names
    """
    feature_names = []
    feature_names += args.bands
    if args.s2_dates : feature_names += ['s2_num_days', 's2_doy_cos', 's2_doy_sin']
    if args.s1 : feature_names += ['s1_vv', 's1_vh', 's1_num_days', 's1_doy_cos', 's1_doy_sin']
    if args.latlon : feature_names += ['lat_cos', 'lat_sin', 'lon_cos', 'lon_sin']
    else: feature_names += ['lat_cos', 'lat_sin']
    if args.gedi_dates : feature_names += ['gedi_num_days', 'gedi_doy_cos', 'gedi_doy_sin']
    if args.alos : feature_names += ['alos_hh', 'alos_hv']
    if args.ch : feature_names += ['ch', 'ch_std']
    if args.lc : feature_names += ['lc_cos', 'lc_sin', 'lc_prob']
    if args.dem : feature_names += ['dem']
    return feature_names


def create_table(fnames, paths, mode, year = 2019) :
    """
    This function creates the features and labels csv files for the train, val and test sets.
    You can run it locally with the following arguments:
        - fnames = [f'data_subset-2019-v3_{i}-20.h5' for i in range(20)]
        - paths = {'h5':'/scratch2/gsialelli/patches', 'norm': '/scratch2/gsialelli/patches/statistics_subset_2019-v3.pkl', 
        'map': '/scratch2/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/biomes_split'}

    Args:
    - fnames : list : the names of the h5 files

    Returns:
    - None
    """

    # Set up arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.latlon = True
    args.bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    args.ch = True
    args.s1 = False
    args.alos = True
    args.lc = True
    args.dem = True
    args.gedi_dates = False
    args.patch_size = [1,1]
    args.norm_strat = 'pct'
    args.norm = False
    args.s2_dates = False
    args.gedi_dates = False

    # Get the feature names
    features_names = construct_feature_names(args)

    # Iterate over the modes
    print(f'Processing {mode} data...')

    # Get the dataset
    custom_dataset = GEDIDataset(paths, fnames = fnames, chunk_size = 1, mode = mode, args = args)
    data_loader = DataLoader(dataset = custom_dataset,
                            batch_size = 1024,
                            shuffle = False,
                            num_workers = 8)

    # Iterate through the DataLoader
    print('starting to iterate...')
    
    for batch_idx, batch_samples in enumerate(tqdm(data_loader)):
        
        features, labels = batch_samples
        features = features.squeeze(2).squeeze(2).numpy().astype(np.float32)
        labels = labels.numpy().astype(np.float32)

        assert features.shape[1] == len(features_names), f'Expected {len(features_names)} features, got {features.shape[1]}'

        if batch_idx == 0 :
            df_features = pd.DataFrame(features, columns = features_names)
            df_labels = pd.DataFrame(labels, columns = ['agbd'])
        else: 
            df_features = pd.concat([df_features, pd.DataFrame(features, columns=features_names)], ignore_index=True)
            df_labels = pd.concat([df_labels, pd.DataFrame(labels, columns=['agbd'])], ignore_index=True)
        
    print('done!')
    print()

    # Save the data
    df_features.to_csv(join(paths['h5'], f'{mode}_features_{year}.csv'), index = False)
    df_labels.to_csv(join(paths['h5'], f'{mode}_labels_{year}.csv'), index = False)


class RF_GEDIDataset(Dataset) :
    """
    This class is a subclass of torch.utils.data.Dataset. It is used to load the features and labels from the csv files.
    """

    def __init__(self, data_path, mode, args, years) :
        """
        This function initializes the class.

        Args:
        - paths: dict, the paths to the data
        - mode: str, the mode of the dataset
        - years: list, the year(s) of the dataset

        Returns:
        - None
        """

        # Get the features to be used
        columns_to_load = construct_feature_names(args)
        if mode == 'train' : print('Loading features:', columns_to_load)

        # Load the features
        features_data = [pd.read_csv(join(data_path, f'{mode}_features_{year}.csv'), usecols = columns_to_load) for year in years]
        self.features = pd.concat(features_data, ignore_index = True)
        
        # Load the labels
        labels_data = [pd.read_csv(join(data_path, f'{mode}_labels_{year}.csv')) for year in years]
        self.labels = pd.concat(labels_data, ignore_index = True)