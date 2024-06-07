## Patches creation procedure

This folder contains the code and data to showcase how the patches were obtained for the dataset. 

First, of all run the following command to download the example data. This includes a handful of Sentinel-2 L2A products, the ALOS-2 PALSAR-2 yearly mosaic for 2020, the JAXA Digital Elevation Model, the Copernicus Land Cover, and the yearly Canopy Height Map for 2020. We take for this example, the Sentinel-2 tile 30NXM, located in Ghana.
```
wget "https://libdrive.ethz.ch/index.php/s/VPio6i5UlXTgir0/download?path=%2F&files=example_data&downloadStartSecret=gxairgqzc" -O example_data.tar
```


Then, one can simply run:
```
python create_patches.py  --tilenames example.txt \
                          --year 2020 \
                          --patch_size 25 25 \
                          --chunk_size 1 \
                          --path_shp S2_index/sentinel_2_index_shapefile.shp \
                          --path_gedi example_data/L4A_30NXM.shp \
                          --path_s2 example_data/ \
                          --path_alos example_data/ \
                          --path_dem example_data/ \
                          --path_ch example_data/ \
                          --path_lc example_data/ \
                          --output_path example_data/ \
                          --output_fname example_patches \
                          --ALOS --CH --LC --DEM \
                          --i 0 --N 1
```

This will create the file `example_data/data-2019_example_patches_0-1.h5`.

A description of the command line arguments can be found in the [code](https://github.com/ghjuliasialelli/AGBD/blob/main/Patches/create_patches.py) itself.
