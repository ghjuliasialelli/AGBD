## Patches creation procedure

This folder contains the code and data to showcase how the patches were obtained for the dataset. 

First, of all run the following command to download the example data. This includes a handful of Sentinel-2 L2A products, the ALOS-2 PALSAR-2 yearly mosaic for 2020, the JAXA Digital Elevation Model, the Copernicus Land Cover, and the yearly Canopy Height Map for 2020. We take for this example, the Sentinel-2 tile 30NXM, located in Ghana.
```
wget 'https://libdrive.ethz.ch/index.php/s/VPio6i5UlXTgir0/download?path=%2F&files=example_data&downloadStartSecret=y3gyn0c105l'
```


Then, one can simply run:
```
python create_patches.py  --tilenames example.txt \                            # file listing the tile(s) for which to extract the patches
                          --year 2020 \                                        # year to consider for the S2 products
                          --patch_size 25 25 \                                 # size for the patches
                          --chunk_size 1 \                                     # hdf5 chunk size
                          --path_shp S2_index/sentinel_2_index_shapefile.shp \ # shapefile for the S2 tiles
                          --path_gedi example_data/L4A_30NXM.shp \             # path to the GEDI data
                          --path_s2 example_data/ \                            # path to the S2 products
                          --path_alos example_data/ \                          # path to the ALOS data
                          --path_dem example_data/ \                           # path to the DEM data
                          --path_ch example_data/ \                            # path to the CH data
                          --path_lc example_data/ \                            # path to the LC data
                          --output_path example_data/ \                        # path to the directory in which to save the output file
                          --output_fname example_patches \                     # name of the output file
                          --ALOS --CH --LC --DEM \                             # flags to indicate which data products to extract
                          --i 0 --N 1                                          # split the listed tile(s) into N chunks, and process the i-th chunk 
```

This will create the file `example_data/data_example_patches_0-1.h5`.

A description of the command line arguments can be found in the [code](https://github.com/ghjuliasialelli/AGBD/blob/main/Patches/create_patches.py) itself.
