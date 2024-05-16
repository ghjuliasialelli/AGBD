"""

Set-up of argument parsing for models training. Defines the `setup_parser()` function, which returns an `ArgumentParser()`
object containing the command-line arguments.

"""

import argparse

def str2bool(v):
    """ 
        Helper function to parse a string into a boolean.
        
        input: `v` (str), input string to be parsed
        output: bool
    """
    if v == 'true': return True
    elif v == 'false': return False
    else: raise argparse.ArgumentTypeError(f"Either 'true' or 'false' expected, got {v}.")


def setup_parser():
    """ 
        Main function. Returns an `ArgumentParser()` object containing the command-line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', required = True, type = str, help = 'Path to the folder where models should be saved.')
    parser.add_argument('--model_name', required = True, type = str, help = 'Model name, used as the .pth filename to save it.')

    # Dataset #################################################################################################################################################
    parser.add_argument("--dataset_path", type = str, required = True, help = 'Path to the dataset.')
    parser.add_argument("--augment", type = str2bool, default = 'true', help = 'Whether to perform data augmentation.')
    parser.add_argument("--norm", type = str2bool, default = 'false', help = 'Whether to normalize the agbd.')
    parser.add_argument("--chunk_size", type = int, default = 1, help = 'Internal chunk size of the hdf5.')

    # Model ###################################################################################################################################################
    
    parser.add_argument("--arch",   required = True, type = str, help = 'Network architecture.')
    parser.add_argument("--model_idx", type = int, help = 'Model ID, within the ensemble.')
    parser.add_argument("--loss_fn", required = True, type = str, help = 'Which loss function to use for the training. Can be: `RMSE`, `GNLL`, `LNLL`. Not considered if `mt_weighting` is `uncertainty`.')

    # inputs
    parser.add_argument("--latlon", type = str2bool, default = 'true', help = 'Whether to include `lon_1` and `lon_2` in the input features.')
    parser.add_argument("--ch", type = str2bool, default = 'false', help = 'Whether to include the `ch` and `ch_std` patches in the input.')
    parser.add_argument("--bands", type = str, nargs = '*', help = 'Sentinel-2 bands (e.g., `B12`) to consider as input for the model.' )
    parser.add_argument("--in_features", required = True, type = int, help = 'Number of features provided as input to the model.')
    parser.add_argument("--s1", type = str2bool, default = 'false', help = 'Whether to include the S1 patches in the input.')
    parser.add_argument("--alos", type = str2bool, default = 'false', help = 'Whether to include the ALOS patches in the input.')
    parser.add_argument("--lc", type = str2bool, default = 'false', help = 'Whether to include the LC patches in the input.')
    parser.add_argument("--dem", type = str2bool, default = 'false', help = 'Whether to include the DEM patches in the input.')
    parser.add_argument("--gedi_dates", type = str2bool, default = 'false', help = 'Whether to include the GEDI dates in the input.')
    parser.add_argument("--s2_dates", type = str2bool, default = 'false', help = 'Whether to include the GEDI dates in the input.')

    # outputs 
    parser.add_argument("--num_outputs", required = True, type = int, help = 'Number of features outputed by the model.')
    parser.add_argument("--norm_strat", type = str, required = True, help = 'Normalization strategy, one of `pct`, `min_max` and `mean_std`.')

    # Training ################################################################################################################################################
    
    # model
    parser.add_argument("--n_epochs", default = 100, type = int, help = 'Number of epochs.')
    parser.add_argument("--limit", type = str2bool, default = 'true', help = 'Whether to limit the number of batches to process at each epoch.')
    parser.add_argument("--batch_size", default = 256, type = int, help= 'Batch size.')
    parser.add_argument("--years", type = int, nargs = '+', help = 'Year of the dataset.')

    # FCN model arguments
    parser.add_argument("--channel_dims", type = int, nargs = '*', help = 'List of channel feature dimensions.')
    parser.add_argument("--downsample", type = str2bool, default = 'true', help = 'Whether to downsample the patches from 10m resolution to 50m resolution.')
    parser.add_argument("--max_pool", type = str2bool, default = 'false', help = 'Whether to use max pooling after each convolutional layer.')
    
    # UNet model arguments
    parser.add_argument("--leaky_relu", type = str2bool, default = 'false', help = 'Whether to use leaky ReLU activation functions.')

    # optimizer & scheduler
    parser.add_argument("--lr", default = 1e-4, type = float, help = 'Learning rate.')
    parser.add_argument("--step_size", default = 30, type = int, help = 'Period of learning rate decay.')
    parser.add_argument("--gamma", default = 0.1, type = float, help = 'Multiplicative factor of learning rate decay.')

    # early stopping
    parser.add_argument("--patience", default = 3, type = int, help = 'Number of checks with no improvements after which training will be stopped.')
    parser.add_argument("--min_delta", default = 0.0, type = float, help = 'Minimum change in the monitored quantity to qualify as improvement.')

    # re-balancing
    parser.add_argument("--reweighting", type = str, help = 'Method to be used for samples weights reweighting.')
    
    # Predict #################################################################################################################################################
    
    parser.add_argument("--tile_name", type = str, help = 'Path to the tile on which to run the prediction.')
    parser.add_argument("--clip", type = str2bool, default = 'true', help = 'Whether to clip AGBD values to the [0, 500] range.')
    parser.add_argument('--output_path', type = str, help = 'Path to the folder where predictions should be saved.')

    # ensemble
    parser.add_argument("--n_models", type = int, help = 'Number of models to train as an ensemble.')
    parser.add_argument('--patch_size', help = 'Size of the patches to extract, in pixels.', nargs = 2, type = int, default = [25, 25])

    return parser
