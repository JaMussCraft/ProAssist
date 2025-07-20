import os
import glob
import subprocess
import multiprocessing

data_dir = "/fsx/work/imzyc/datasets/EPIC-KITCHEN-55/frames_rgb_flow/rgb/*/*/*" # for EK55
# data_dir = "/fsx/work/imzyc/datasets/EPIC-KITCHEN-100/*/rgb_frames/*" # for EK100

# iterate over all the files in the directory
cnt = 0
all_files = [file for file in glob.glob(data_dir) if '.tar' in file]
all_files = sorted(all_files)

def extract_frames(file):
    output_dir = file.replace('.tar', '')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        # unzip .tar
        print(f"Extracting {file}...")
        subprocess.run(f"tar -xf {file} -C {output_dir}", shell=True)

num_proc = 32
mp = multiprocessing.Pool(num_proc)
mp.map(extract_frames, all_files)