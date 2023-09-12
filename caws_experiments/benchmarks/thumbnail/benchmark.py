from caws.task import caws_task
from caws.features import ArgAttrFeature

def thumbnailer(img_path, width, height, *, _caws_output_dir):
    import datetime
    import io
    import os
    import sys
    import uuid
    from urllib.parse import unquote_plus
    from PIL import Image
    
    os.makedirs(_caws_output_dir, exist_ok=True)

    # Memory-based solution
    def resize_image(img_path, w, h, output_dir):
        with Image.open(img_path) as image:
            image.thumbnail((w,h))
            out = os.path.join(output_dir, "thumbnail.jpeg")
            image.save(out, format='jpeg')
            return out

    process_begin = datetime.datetime.now()
    resized = resize_image(img_path, width, height, _caws_output_dir)
    resized_size = os.path.getsize(resized)
    process_end = datetime.datetime.now()

    process_time = (process_end - process_begin) / datetime.timedelta(microseconds=1)

    return {
            'result': {
                'path': resized
            },
            'measurement': {
                'upload_size': resized_size,
                'compute_time': process_time
            }
    }

thumbnailer = caws_task(thumbnailer, features=[ArgAttrFeature("size")])