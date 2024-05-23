## Patches creation procedure

This folder contains the code and data to showcase how the patches were obtained for the dataset. 

We take for this example, the Sentinel-2 tile TBD, located in TBD. We provide: a handful of Sentinel-2 L2A products, the ALOS-2 PALSAR-2 yearly mosaic for 2020, the JAXA Digital Elevation Model, the Copernicus Land Cover, and the yearly Canopy Height Map for 2020.
One can simply run:
```
python create_patches.py  --tilenames example_TBD.txt
                          --year 2019
                          --patch_size 25 25
                          --chunk_size 1
                          --path_shp S2_index/sentinel_2_index_shapefile.shp
                          --path_gedi TBD.shp
                          --path_s2 ./
                          --path_alos ./
                          --path_ch ./
                          --path_lc ./
                          --output_path ./
                          --output_fname example_patches
                          --ALOS --CH --LC --DEM --i 0 --N 1
```

A description of the command line arguments can be found in the [code](https://github.com/ghjuliasialelli/AGBD/blob/main/Data/create_patches.py) itself.
