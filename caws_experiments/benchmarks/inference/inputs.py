from caws.path import CawsPath

def generate_inputs(src_endpoint, size, data_dir=""):
    # upload model
    data_dir = os.path.join(data_dir, "400.inference", "411.image-recognition")

    model_name = 'resnet50-19c8e357.pth'
    model_path = CawsPath(src_endpoint, os.path.join(data_dir, 'model', model_name))

    SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    class_index_path = CawsPath(src_endpoint, os.path.join(SCRIPT_DIR, "imagenet_class_index.json"))

    input_images = []
    resnet_path = os.path.join(data_dir, 'fake-resnet')
    with open(os.path.join(resnet_path, 'val_map.txt'), 'r') as f:
        for line in f:
            img, img_class = line.split()
            input_images.append((img, img_class))
            image_path = CawsPath(src_endpoint, os.path.join(resnet_path, img), isolate=True)
            break # Only need one image
    
    return (model_path, image_path, class_index_path), {}