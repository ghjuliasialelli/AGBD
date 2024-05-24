from glob import glob
import os

for sh in glob("*/*.sh"):
    os.system(f"sbatch {sh}")
