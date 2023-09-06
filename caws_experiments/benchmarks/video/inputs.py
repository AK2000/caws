from caws.path import CawsPath

def generate_inputs(src_endpoint, size, data_dir=""):
    for file in glob.glob(os.path.join(data_dir, "200.multimedia", "220.video-processing", '*.mp4')):
        img = os.path.relpath(file, data_dir)
        img_path = CawsPath(src_endpoint, img, isolate=True)

    watermark_path = CawsPath(src_endpoint, os.path.join(data_dir, "resources", "watermark.png"))
    return (img_path, "watermark"), {"watermark_path": watermark_path}