## Patches creation procedure

This folder contains the code and data to showcase how the patches were obtained for the dataset. 

First, of all run the following command to download the example data. This includes a handful of Sentinel-2 L2A products, the ALOS-2 PALSAR-2 yearly mosaic for 2020, the JAXA Digital Elevation Model, the Copernicus Land Cover, and the yearly Canopy Height Map for 2020. We take for this example, the Sentinel-2 tile 30NXM, located in Ghana.
```
wget https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/674193/example_data
```


One can simply run:
```
python create_patches.py  --tilenames example_TBD.txt
                          --year 2019
                          --patch_size 25 25
                          --chunk_size 1
                          --path_shp S2_index/sentinel_2_index_shapefile.shp
                          --path_gedi TBD.shp
                          --path_s2 example_data/
                          --path_alos example_data/
                          --path_ch example_data/
                          --path_lc example_data/
                          --output_path ./
                          --output_fname example_patches
                          --ALOS --CH --LC --DEM --i 0 --N 1
```

A description of the command line arguments can be found in the [code](https://github.com/ghjuliasialelli/AGBD/blob/main/Data/create_patches.py) itself.