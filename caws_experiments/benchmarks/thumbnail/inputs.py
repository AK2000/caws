from caws.path import CawsPath

def generate_input(src_endpoint, size, data_dir=""):
    for file in glob.glob(os.path.join(data_dir, "200.multimedia", "210.thumbnailer", '*.jpg')):
        img = os.path.relpath(file, data_dir)
        img_path = CawsPath(src_endpoint, img, isolate=True)
        break # We only use one image, not sure why SeBS "uploads" them all
        
    return (img_path, 200, 200), {}