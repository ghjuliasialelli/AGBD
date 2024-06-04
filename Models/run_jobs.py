"""

This script runs all the scripts in the scripts/ folder.

"""

from glob import glob
import os

for sh in glob("scripts/*/*.sh"):
    os.system(f"sbatch {sh}")
