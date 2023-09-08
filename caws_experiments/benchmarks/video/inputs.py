import glob
import os

from caws.path import CawsPath

def generate_inputs(src_endpoint, size, data_dir=""):
    for file in glob.glob(os.path.join(data_dir, "200.multimedia", "220.video-processing", '*.mp4')):
        img_path = CawsPath(src_endpoint, file, isolate=True)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    watermark_path = CawsPath(src_endpoint, os.path.join(script_dir, "resources", "watermark.png"))
    return (img_path, 1, "watermark"), {"watermark_path": watermark_path}