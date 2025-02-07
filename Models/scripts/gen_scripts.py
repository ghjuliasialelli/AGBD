"""

This script is used to generate the bash scripts to launch the training of the baseline models.
The scripts will be saved as /runs/{architecture}_{patch_size}/{config}.sh, where config describes the configuration of the model.

"""


############################################################################################################################
# IMPORTS ##################################################################################################################
############################################################################################################################

from os.path import isdir, join
from os import makedirs
from string import Template
import re

############################################################################################################################
# EXECUTION ################################################################################################################
############################################################################################################################

# Read template content
with open("template.sh", "r") as f:
    template_content = f.read()

# Architectures and patch sizes to loop over
architectures = ['fcn', 'unet', 'nico', 'rf']
patch_sizes = ['15', '25']

configs = {'all' : {'in_ch': 'true', 'in_bands': 'all', 'in_latlon': 'true', 'in_alos': 'true', 'in_lc': 'true', 'in_dem': 'true'},
            'ch':{'in_ch': 'true', 'in_bands': 'no', 'in_latlon': 'true', 'in_alos': 'false', 'in_lc': 'false', 'in_dem': 'false'},
            'RGBN':{'in_ch': 'false', 'in_bands': 'rgbn', 'in_latlon': 'true', 'in_alos': 'false', 'in_lc': 'false', 'in_dem': 'false'},
            'S2':{'in_ch': 'false', 'in_bands': 'all', 'in_latlon': 'true', 'in_alos': 'false', 'in_lc': 'false', 'in_dem': 'false'},
            'S2ALOS':{'in_ch': 'false', 'in_bands': 'all', 'in_latlon': 'true', 'in_alos': 'true', 'in_lc': 'false', 'in_dem': 'false'},
            'S2DEM':{'in_ch': 'false', 'in_bands': 'all', 'in_latlon': 'true', 'in_alos': 'false', 'in_lc': 'false', 'in_dem': 'true'},
            'S2LC':{'in_ch': 'false', 'in_bands': 'all', 'in_latlon': 'true', 'in_alos': 'false', 'in_lc': 'true', 'in_dem': 'false'},
            'S2LCDEMALOS':{'in_ch': 'false', 'in_bands': 'all', 'in_latlon': 'true', 'in_alos': 'true', 'in_lc': 'true', 'in_dem': 'true'}
            }

for arch in architectures:

    # Define mem_req
    if arch == 'fcn' :
        epochs = "25"
        mem_req = "#SBATCH --gpus=1\n#SBATCH --gres=gpumem:10g"
        time_req = "#SBATCH --time=72:00:00"
    elif arch == 'unet' :
        epochs = "10"
        mem_req = "#SBATCH --gpus=1\n#SBATCH --gres=gpumem:10g"
        time_req = "#SBATCH --time=72:00:00"
    else:
        epochs = "10"
        mem_req = "#SBATCH --gpus=1\n#SBATCH --gres=gpumem:24g"
        time_req = "#SBATCH --time=120:00:00"

    if arch == 'rf': patch_sizes = ['1']

    for ps in patch_sizes:
    
        if not isdir(f"{arch}_{ps}"): makedirs(f"{arch}_{ps}")

        for cf_name, config in configs.items():

            # Supplement the config with the architecture, patch size, and memory requirement
            config['in_mem_req'] = mem_req
            config['in_time_req'] = time_req
            config['in_epochs'] = epochs
            config['in_arch'] = arch
            config['in_ps'] = ps

            # Iterate over the variables to replace in the template
            file_content = template_content[:]
            for k, v in config.items():
                pattern = re.compile(re.escape(k))
                # Don't want quotes for in_mem_req or in_ps
                if k in ['in_mem_req', 'in_ps', 'in_time_req', 'in_epochs']:
                    file_content = re.sub(pattern, v, file_content)
                else:
                    file_content = re.sub(pattern, '"' + v + '"', file_content)

            # Write to a new file
            output_file = join(f'{arch}_{ps}', f'train_{cf_name}.sh')
            with open(output_file, 'w') as f:
                f.write(file_content)
        

