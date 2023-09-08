import glob
import os

from caws.path import CawsPath

def generate_inputs(src_endpoint, size, data_dir=""):
    for file in glob.glob(os.path.join(data_dir, "200.multimedia", "210.thumbnailer", '*.jpg')):
        img_path = CawsPath(src_endpoint, file, isolate=True)
        break # We only use one image, not sure why SeBS "uploads" them all
        
    return (img_path, 200, 200), {}