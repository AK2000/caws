from caws.task import caws_task

idx2label = None
model = None

def image_recognition(model_path, image_path, class_index_path):
    import datetime, json, os, uuid
    from PIL import Image
    import torch
    from torchvision import transforms
    from torchvision.models import resnet50

    global idx2label
    if not idx2label:
        class_idx = json.load(open(os.path.join(class_index_path), 'r'))
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

    global model
    if not model:
        model_process_begin = datetime.datetime.now()
        model = resnet50(pretrained=False)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model_process_end = datetime.datetime.now()
    else:
        model_process_begin = datetime.datetime.now()
        model_process_end = datetime.datetime.now()
    
    process_begin = datetime.datetime.now()
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model 
    output = model(input_batch)
    _, index = torch.max(output, 1)
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    prob = torch.nn.functional.softmax(output[0], dim=0)
    _, indices = torch.sort(output, descending = True)
    ret = idx2label[index]
    process_end = datetime.datetime.now()
    
    model_process_time = (model_process_end - model_process_begin) / datetime.timedelta(microseconds=1)
    process_time = (process_end - process_begin) / datetime.timedelta(microseconds=1)
    return {
            'result': {'idx': index.item(), 'class': ret},
            'measurement': {
                'compute_time': process_time + model_process_time,
                'model_time': model_process_time,
            }
        }

inference = caws_task(inference)