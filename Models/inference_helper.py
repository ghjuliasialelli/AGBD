"""

This script contains the helper functions used in the inference process.

"""

#######################################################################################################################
# Imports

import numpy as np
import rasterio as rs
import xml.etree.ElementTree as ET
import datetime as dt
import glob
from os.path import join
from rasterio.crs import CRS
from rasterio.transform import AffineTransformer
from scipy.ndimage import distance_transform_edt
from skimage.transform import resize
from pyproj import Transformer
from os.path import exists


# Sentinel-2 L2A bands that we want to use
S2_L2A_BANDS = {'10m' : ['B02', 'B03', 'B04', 'B08'],
                '20m' : ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SCL'],
                '60m' : ['B01', 'B09']}

# Sentinel-2 attributes and their corresponding data types (we differentiate between 2d and 1d attributes)
S2_attrs = {'bands' : {'B01': np.uint16, 'B02': np.uint16, 'B03': np.uint16, 'B04': np.uint16, 'B05': np.uint16, 'B06': np.uint16, 
                        'B07': np.uint16, 'B08': np.uint16, 'B8A': np.uint16, 'B09': np.uint16, 'B11': np.uint16, 'B12': np.uint16, 
                        'SCL': np.uint8},
            'metadata' : {'vegetation_score': np.uint8, 'date' : np.int16, 'pbn' : np.uint16, 'ron' : np.uint8, 'boa_offset': np.uint8}
            }

# ALOS PALSAR-2 attributes and their corresponding data types
ALOS_attrs = {'HH': np.uint16, 'HV': np.uint16}

# Canopy Height (CH) attributes and their corresponding data types
CH_attrs = {'ch': np.uint8, 'std': np.uint8}

# Land Cover (LC) attributes and their corresponding data types
LC_attrs = {'lc': np.uint8, 'prob' : np.uint8}

# ALOS DEM attributes and their corresponding data types
DEM_attrs = {'dem': np.int16}

NODATAVALS = {'S2' : 0, 'CH': 255, 'ALOS': 0, 'LC': 255, 'DEM': -9999, 'LC': 255}

#######################################################################################################################
# Helper functions 


def crop_and_pad_arrays(data, ul_row, lr_row, ul_col, lr_col, invalid):
    """
    This function crops (and pads if necessary) the data to match the shape provided by
    the upper left and lower right indices.

    Args:
    - data: 2d array, data.
    - (ul_row, ul_col) : tuple of ints, indices of the pixel corresponding to the upper
        left corner of the Sentinel-2 tile.
    - (lr_row, lr_col) : tuple of ints, indices of the pixel corresponding to the lower
        right corner of the Sentinel-2 tile.
    - invalid: int/float, value to use for the padding.
    
    Returns:
    - data: 2d array, cropped (and padded) data.
    """

    # Get the dimensions of the arrays
    height, width = data.shape

    # If any of the slicing indices are out of bounds, pad with zeros
    if ul_row < 0 or lr_row >= height or ul_col < 0 or lr_col >= width:

        print('(padding)')

        # Calculate the new shape after padding
        new_height = lr_row - ul_row + 1
        new_width = lr_col - ul_col + 1

        # Create new arrays to store the padded data
        padded_data = np.full(shape = (new_height, new_width), fill_value = invalid, dtype = data.dtype)

        # Compute the region of interest in the new padded arrays
        start_row = max(0, -ul_row)
        end_row = min(height - ul_row, lr_row - ul_row + 1)
        start_col = max(0, -ul_col)
        end_col = min(width - ul_col, lr_col - ul_col + 1)

        # Copy the original data to the new padded arrays
        padded_data[start_row : end_row, start_col : end_col] = data[max(0, ul_row) : min(height, lr_row + 1), max(0, ul_col) : min(width, lr_col + 1)]

        # Update the variables to point to the new padded arrays
        data = padded_data

    # Otherwise, simply perform the slicing operation
    else: data = data[ul_row : lr_row + 1, ul_col : lr_col + 1]

    return data


def fill_nan_with_nearest(image, nan_mask):
    """
    This function fills the NaN values in the image with the nearest non-NaN value.

    Args:
    - image: 2d array, image with NaN values.
    - nan_mask: 2d array, mask of the NaN values in the image.

    Returns:
    - filled_image: 2d array, image with NaN values filled.
    """
    
    indices = distance_transform_edt(nan_mask, return_distances = False, return_indices = True)
    filled_image = image[tuple(indices)]
    
    return filled_image


def upsampling_with_nans(image, upsampling_shape, nan_value, order) :
    """
    This function upsamples the image to the `upsampling_shape`, and fills the NaN values with the nearest non-NaN value.

    Args:
    - image: 2d array, image to upsample.
    - upsampling_shape: tuple of ints, shape of the upsampled image.
    - nan_value: int, value to use for the NaN values.
    - order: int, order of the interpolation.
        order = 0 : nearest neighbor interpolation
        order = 1 : bilinear interpolation
        order = 2 : bi-quadratic interpolation
        order = 3 : bicubic interpolation
        cf. https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp

    Returns:
    - upsampled_image_with_nans: 2d array, upsampled image with NaN values filled.
    """

    # Check that there are no inf values in the data
    assert not np.isinf(image).any(), 'There are inf values in the data.'

    # Create a mask for the non-defined values
    if np.isnan(nan_value) : nan_mask = np.isnan(image)
    else: nan_mask = (image == nan_value)

    # If there are no undefined values, simply resize
    if np.count_nonzero(nan_mask) == 0 :
        return resize(image, upsampling_shape, order = order, mode = 'edge', preserve_range = True)

    # Otherwise, take care of the undefined values
    else:

        # In the original image, fill the NaN values with the nearest non-NaN value
        non_nan_image = fill_nan_with_nearest(image, nan_mask)

        # Upsample the original image
        upsampled_image = resize(non_nan_image, upsampling_shape, order = order, mode = 'edge', preserve_range = True)

        # Upsample the NaN mask
        upsampled_nan_mask = resize(nan_mask.astype(float), upsampling_shape, order = 0, mode = 'edge') > 0.5

        # Replace the NaN values in the upsampled image with NaN
        upsampled_image_with_nans = np.where(upsampled_nan_mask, nan_value, upsampled_image)

        return upsampled_image_with_nans


def get_tile(data, s2_transform, upsampling_shape, data_source, data_attrs) :
    """
    This function extracts the data for the Sentinel-2 L2A product at hand, crops it so as to perfectly match
    the Sentinel-2 tile, resamples it to 10m resolution when necessary, and returns it.

    Args:
    - data: dict, with the attributes as keys and the corresponding 2d arrays as values.
    - s2_transform: affine.Affine, transform of the Sentinel-2 L2A product.
    - upsampling_shape: tuple of ints, shape of the Sentinel-2 L2A product, at 10m resolution.
    - data_source: string, data source.
    - data_attrs: dict, with the attributes as keys and the corresponding data types as values.

    Returns:
    - res: dict, with the attributes as keys and the corresponding 2d arrays as values.
    """

    if data == {} : return None

    # Get the transforms
    s2_transformer, data_transformer = AffineTransformer(s2_transform), AffineTransformer(data['transform'])    

    # Upper left corner
    ul_x, ul_y = s2_transformer.xy(0, 0)
    ul_row, ul_col = data_transformer.rowcol(ul_x, ul_y)

    # Lower right corner
    lr_x, lr_y = s2_transformer.xy(upsampling_shape[0] - 1, upsampling_shape[1] - 1)
    lr_row, lr_col = data_transformer.rowcol(lr_x, lr_y)

    # Crop the data to the same bounds, padding the data if necessary
    res = {}
    for data_attr in data_attrs.keys() :
        res[data_attr] = crop_and_pad_arrays(data[data_attr], ul_row, lr_row, ul_col, lr_col, invalid = 0 if data_source == 'ALOS' else NODATAVALS[data_source])

        # Resample to 10m resolution if necessary, i.e. for data sources with resolution lower than 10m per pixel
    
        if data_source == 'ALOS' :
            res[data_attr] = upsampling_with_nans(res[data_attr].astype(np.float32), upsampling_shape, NODATAVALS[data_source], 1).astype(data_attrs[data_attr])
        
        if data_source == 'DEM' :
            res[data_attr] = upsampling_with_nans(res[data_attr].astype(np.float32), upsampling_shape, NODATAVALS[data_source], 1).astype(data_attrs[data_attr])
        
        elif data_source == 'LC' :
            res[data_attr] = upsampling_with_nans(res[data_attr], upsampling_shape, NODATAVALS[data_source], 0).astype(data_attrs[data_attr])
        
        assert res[data_attr].shape == upsampling_shape, f'{data_source} | {data_attr} | {data[data_attr].shape} | {res[data_attr].shape} | {upsampling_shape} | {ul_row} | {lr_row} | {ul_col} | {lr_col}'

    return res


def radiometric_offset_values(path_s2, product, offset) :
    """
    This function extracts the BOA_QUANTIFICATION_VALUE and BOA_ADD_OFFSET_VALUES from the
    Sentinel-2 L2A product at hand, and returns them.

    Args:
    - path_s2: string, path to the Sentinel-2 data directory.
    - product: string, name of the Sentinel-2 L2A product.
    - offset: int, 1 if the product was acquired after January 25th, 2022; 0 otherwise.

    Returns:
    - None
    """

    # There is a mismatch between the names of the physical bands in the metadata file, and the
    # names of the bands in the IMG_DATA/ folder. This dictionary defines the mapping
    bands_mapping = {'B1': 'B01', 'B2': 'B02', 'B3': 'B03', 'B4': 'B04', 'B5': 'B05', 'B6': 'B06', 'B7': 'B07', \
                    'B8': 'B08', 'B8A': 'B8A', 'B9': 'B09', 'B10': 'B10', 'B11': 'B11', 'B12': 'B12'}

    # Parse the XML file
    tree = ET.parse(f'{join(path_s2, product)}.SAFE/MTD_MSIL2A.xml')
    root = tree.getroot()

    # Get the BOA_QUANTIFICATION_VALUE
    for elem in root.find('.//QUANTIFICATION_VALUES_LIST') :
        if elem.tag == 'BOA_QUANTIFICATION_VALUE' :
            boa_quantification_value = float(elem.text)
            assert boa_quantification_value == 10000, f'BOA_QUANTIFICATION_VALUE is {boa_quantification_value}, should be 10000'
        else: continue

    # Get the physical bands and their ids
    physical_bands = {elem.get('bandId'): elem.get('physicalBand') \
                      for elem in root.find('.//Spectral_Information_List')}

    if offset :
        
        # Check the BOA offset values (should be 1000)
        for elem in root.find('.//BOA_ADD_OFFSET_VALUES_LIST') :
            physical_band = physical_bands[elem.get('band_id')]
            actual_band = bands_mapping[physical_band]
            boa_add_offset_value = int(elem.text)
            assert boa_add_offset_value == 1000, f'BOA_ADD_OFFSET_VALUE is {boa_add_offset_value}, should be 1000 | band {actual_band}'


def encode_tile(tile_reader, transformer) :
    """ 
    This function encodes the lat/lon of the tile in the [0,1] range.

    Args:
    - tile_reader: rasterio dataset, tile reader.
    - transformer: pyproj.Transformer, transformer.

    Returns:
    - lat_cos, lat_sin, lon_cos, lon_sin: 2d arrays, lat/lon in the [0,1] range.
    """

    width, height = tile_reader.width, tile_reader.height
    top, bottom, right = tile_reader.xy(0,0), tile_reader.xy(height, 0), tile_reader.xy(0, width)
    top, bottom, right = [transformer.transform(x,y) for (x,y) in [top, bottom, right]]
    
    # For longitude, we only need to calculate the first row, which we do by interpolating
    dist = np.abs(top[0] - right[0])
    incr = dist / (width - 1)
    row = np.append(np.arange(start = top[0], stop = right[0], step = incr), right[0])[:10980]

    # For latitude, we only need to calculate the first column, which we do by interpolating
    dist = np.abs(top[1] - bottom[1]) 
    incr = dist / (height - 1)
    column = np.append(np.arange(start = top[1], stop = bottom[1], step = incr), bottom[1])[:10980]

    # Now we duplicate the relevant row and column to have the desired shape
    lat, lon = np.zeros((height, width)), np.zeros((height, width))
    for i in range(width): lat[:, i] = column
    for i in range(height) : lon[i, :] = row
    
    # The latitude goes from -90 to 90
    lat_cos, lat_sin = np.cos(np.pi * lat / 90), np.sin(np.pi * lat / 90)
    # The longitude goes from -180 to 180
    lon_cos, lon_sin = np.cos(np.pi * lon / 180), np.sin(np.pi * lon / 180)

    # Put everything in the [0,1] range
    lat_cos, lat_sin = (lat_cos + 1) / 2, (lat_sin + 1) / 2
    lon_cos, lon_sin = (lon_cos + 1) / 2, (lon_sin + 1) / 2

    return lat_cos, lat_sin, lon_cos, lon_sin 

def load_LC_data(path_lc, tile_name) :
    """
    This function loads the LC data for the current tile.

    Args:
    - path_lc: string, path to the LC data directory.
    - tile_name: string, name of the Sentinel-2 tile.

    Returns:
    - LC: dictionary, with the LC attributes as keys, and the corresponding values as values.
    """

    LC = {}
    with rs.open(join(path_lc, f'LC_{tile_name}_2019.tif'), 'r') as src :
        LC['lc'] = src.read(1)
        LC['prob'] = src.read(2)
        LC['transform'] = src.transform
    return LC


def load_DEM_data(path_dem, tile_name) :
    """
    This function loads the DEM data for the current tile.

    Args:
    - path_dem: string, path to the DEM data directory.
    - tile_name: string, name of the Sentinel-2 tile.

    Returns:
    - DEM: dictionary, with the DEM attributes as keys, and the corresponding values as values.
    """

    DEM = {}
    with rs.open(join(path_dem, f'DEM_{tile_name}.tif'), 'r') as src :
        DEM['dem'] = src.read(1)
        DEM['transform'] = src.transform
    return DEM


def load_CH_data(path_ch, tile_name, year) :
    """
    This function loads the CH data for the current tile and year.

    Args:
    - path_ch: string, path to the CH data directory.
    - tile_name: string, name of the Sentinel-2 tile.
    - year: str, year of the Sentinel-2 product.

    Returns:
    - CH: dictionary, with the CH attributes as keys, and the corresponding values as values.
    """

    CH = {}

    with rs.open(join(path_ch, tile_name, year, 'preds_inv_var_mean', f'{tile_name}_pred.tif')) as src:
        CH['ch'] = src.read(1)
        CH['transform'] = src.transform

    with rs.open(join(path_ch, tile_name, year, 'preds_inv_var_mean', f'{tile_name}_std.tif')) as src:
        CH['std'] = src.read(1)
    
    return CH


def load_ALOS_data(tile_name, path_alos, year) :
    """
    This function loads the ALOS PALSAR data for the current tile and year.

    Args:
    - tile_name: string, name of the Sentinel-2 tile.
    - path_alos: string, path to the ALOS PALSAR data directory.
    - year: int, year of the Sentinel-2 product.

    Returns:
    - alos_tiles: dictionary, with the years spanned as keys, and the corresponding ALOS PALSAR data as values.
    """

    cropped_year = str(year)[2:4]

    alos_tiles_year = {}
    if exists(join(path_alos, f'ALOS_{tile_name}_{cropped_year}.tif')) :
        
        with rs.open(join(path_alos, f'ALOS_{tile_name}_{cropped_year}.tif'), 'r') as dataset:
            alos_tiles_year['transform'] = dataset.transform
            band_1, band_2 = dataset.descriptions
            alos_tiles_year[band_1] = dataset.read(1)
            alos_tiles_year[band_2] = dataset.read(2)
    
    return alos_tiles_year


def process_S2_tile(product, path_s2) :
    """
    This function iterates over the bands of the Sentinel-2 L2A product at hand; reprojects them to
    EPSG 4326; upsamples them to 10m resolution (when needed) using bi-linear interpolation (nearest
    neighbor for the scene classification mask); and returns them.
    
    Args:
    - product: string, name of the Sentinel-2 L2A product.
    - path_s2: string, path to the Sentinel-2 data directory.

    Returns:
    - _transform: affine.Affine, transform of the 10m resolution B02 band.
    - upsampling_shape: tuple of ints, shape of the 10m resolution bands.
    - processed_bands: dict, with the bands as keys and the corresponding 2d arrays as values.
    - crs: rasterio.crs.CRS, crs of the bands.
    - bounds: tuple of floats, bounds of the bands.
    - boa_offset: int, 1 if the product was acquired after January 25th, 2022; 0 otherwise.
    - lat_cos, lat_sin, lon_cos, lon_sin: 2d arrays, lat/lon in the [0,1] range.
    - meta: dict, metadata of the bands.
    """

    # Get the path to the IMG_DATA/ folder of the Sentinel-2 product
    path_to_img_data = glob.glob(join(path_s2, product + '.SAFE', 'GRANULE', '*', 'IMG_DATA'))[0]

    # Get the date and tile name from the L2A product name
    _, _, date, _, _, tname, _ = product.split('_')
    year, month, day = int(date[:4]), int(date[4:6]), int(date[6:8])

    # Check the BOA quantification value (and BOA offsets if applicable)
    if dt.date(2022, 1, 25) <= dt.date(year, month, day) : boa_offset = 1 
    else: boa_offset = 0
    radiometric_offset_values(path_s2, product, boa_offset)

    # Iterate over the bands    
    processed_bands = {}
    for res, bands in S2_L2A_BANDS.items() :
        for band in bands :

            # Read the band data
            with rs.open(join(path_to_img_data, f'R{res}', f'{tname}_{date}_{band}_{res}.tif')) as src :
                band_data = src.read(1)
                transform = src.transform
                crs = src.crs
                bounds = src.bounds

                # Extract the lat/lon for one of the bands
                if band == 'B02' :
                    transformer = Transformer.from_crs(crs, 'EPSG:4326')
                    lat_cos, lat_sin, lon_cos, lon_sin = encode_tile(src, transformer)
                    meta = src.meta

            # Turn the band into a 2d array
            if len(band_data.shape) == 3 : band_data = band_data[0, :, :]

            # Use the 10m resolution B02 band as reference
            if res == '10m' :
                if band == 'B02' :
                    # Base the other bands' upsampling on this band's
                    upsampling_shape = band_data.shape
                    # Save the transform of this band
                    _transform = transform
            
            # Upsample the band to a 10m resolution if necessary
            else :
                # Order 0 indicates nearest interpolation, and order 1 indicates bi-linear interpolation
                if band == 'SCL' :
                    band_data = upsampling_with_nans(band_data, upsampling_shape, NODATAVALS['S2'], 0).astype(S2_attrs['bands'][band])
                else:
                    band_data = upsampling_with_nans(band_data.astype(np.float32), upsampling_shape, NODATAVALS['S2'], 1).astype(S2_attrs['bands'][band])

            # Store the transformed band
            assert band_data.shape == (upsampling_shape[0], upsampling_shape[1])
            processed_bands[band] = band_data

    return _transform, upsampling_shape, processed_bands, crs, bounds, boa_offset, lat_cos, lat_sin, lon_cos, lon_sin, meta