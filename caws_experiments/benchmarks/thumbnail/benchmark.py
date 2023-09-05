def thumbnailer(img_path, width, height):
    import datetime
    import io
    import os
    import sys
    import uuid
    from urllib.parse import unquote_plus
    from PIL import Image

    # Memory-based solution
    def resize_image(img_path, w, h):
        with Image.open(img_path) as image:
            image.thumbnail((w,h))
            out = io.BytesIO()
            image.save(out, format='jpeg')
            # necessary to rewind to the beginning of the buffer
            out.seek(0)
            return out

    resized = resize_image(img_path, width, height)
    resized_size = resized.getbuffer().nbytes

    # TODO: Figure out upload of output image
    upload_begin = datetime.datetime.now()
    key_name = client.upload_stream(output_bucket, key, resized)
    upload_end = datetime.datetime.now()

    return resized #TODO: Figure out if this works?