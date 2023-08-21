def thumbnailer():
    import datetime
    import io
    import os
    import sys
    import uuid
    from urllib.parse import unquote_plus
    from PIL import Image

    # Memory-based solution
    def resize_image(image_bytes, w, h):
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.thumbnail((w,h))
            out = io.BytesIO()
            image.save(out, format='jpeg')
            # necessary to rewind to the beginning of the buffer
            out.seek(0)
            return out

    input_bucket = event.get('bucket').get('input')
    output_bucket = event.get('bucket').get('output')
    key = unquote_plus(event.get('object').get('key'))
    width = event.get('object').get('width')
    height = event.get('object').get('height')
    # UUID to handle multiple calls
    #download_path = '/tmp/{}-{}'.format(uuid.uuid4(), key)
    #upload_path = '/tmp/resized-{}'.format(key)
    #client.download(input_bucket, key, download_path)
    #resize_image(download_path, upload_path, width, height)
    #client.upload(output_bucket, key, upload_path)
    download_begin = datetime.datetime.now()
    img = client.download_stream(input_bucket, key)
    download_end = datetime.datetime.now()

    process_begin = datetime.datetime.now()
    resized = resize_image(img, width, height)
    resized_size = resized.getbuffer().nbytes
    process_end = datetime.datetime.now()

    upload_begin = datetime.datetime.now()
    key_name = client.upload_stream(output_bucket, key, resized)
    upload_end = datetime.datetime.now()

    download_time = (download_end - download_begin) / datetime.timedelta(microseconds=1)
    upload_time = (upload_end - upload_begin) / datetime.timedelta(microseconds=1)
    process_time = (process_end - process_begin) / datetime.timedelta(microseconds=1)
    return {
            'result': {
                'bucket': output_bucket,
                'key': key_name
            },
            'measurement': {
                'download_time': download_time,
                'download_size': len(img),
                'upload_time': upload_time,
                'upload_size': resized_size,
                'compute_time': process_time
            }
    }

def video_processing():
    import datetime
    import os
    import stat
    import subprocess


    from . import storage
    client = storage.storage.get_instance()

    SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    def call_ffmpeg(args):
        ret = subprocess.run([os.path.join(SCRIPT_DIR, 'ffmpeg', 'ffmpeg'), '-y'] + args,
                #subprocess might inherit Lambda's input for some reason
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        if ret.returncode != 0:
            print('Invocation of ffmpeg failed!')
            print('Out: ', ret.stdout.decode('utf-8'))
            raise RuntimeError()

    # https://superuser.com/questions/556029/how-do-i-convert-a-video-to-gif-using-ffmpeg-with-reasonable-quality
    def to_gif(video, duration, event):
        output = '/tmp/processed-{}.gif'.format(os.path.basename(video))
        call_ffmpeg(["-i", video,
            "-t",
            "{0}".format(duration),
            "-vf",
            "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop", "0",
            output])
        return output

    # https://devopstar.com/2019/01/28/serverless-watermark-using-aws-lambda-layers-ffmpeg/
    def watermark(video, duration, event):
        output = '/tmp/processed-{}'.format(os.path.basename(video))
        watermark_file = os.path.dirname(os.path.realpath(__file__))
        call_ffmpeg([
            "-i", video,
            "-i", os.path.join(watermark_file, os.path.join('resources', 'watermark.png')),
            "-t", "{0}".format(duration),
            "-filter_complex", "overlay=main_w/2-overlay_w/2:main_h/2-overlay_h/2",
            output])
        return output

    def transcode_mp3(video, duration, event):
        pass

    operations = { 'transcode' : transcode_mp3, 'extract-gif' : to_gif, 'watermark' : watermark }

    def handler(event):
        input_bucket = event.get('bucket').get('input')
        output_bucket = event.get('bucket').get('output')
        key = event.get('object').get('key')
        duration = event.get('object').get('duration')
        op = event.get('object').get('op')
        download_path = '/tmp/{}'.format(key)

        # Restore executable permission
        ffmpeg_binary = os.path.join(SCRIPT_DIR, 'ffmpeg', 'ffmpeg')
        # needed on Azure but read-only filesystem on AWS
        try:
            st = os.stat(ffmpeg_binary)
            os.chmod(ffmpeg_binary, st.st_mode | stat.S_IEXEC)
        except OSError:
            pass

        download_begin = datetime.datetime.now()
        client.download(input_bucket, key, download_path)
        download_size = os.path.getsize(download_path)
        download_stop = datetime.datetime.now()

        process_begin = datetime.datetime.now()
        upload_path = operations[op](download_path, duration, event)
        process_end = datetime.datetime.now()

        upload_begin = datetime.datetime.now()
        filename = os.path.basename(upload_path)
        upload_size = os.path.getsize(upload_path)
        client.upload(output_bucket, filename, upload_path)
        upload_stop = datetime.datetime.now()

        download_time = (download_stop - download_begin) / datetime.timedelta(microseconds=1)
        upload_time = (upload_stop - upload_begin) / datetime.timedelta(microseconds=1)
        process_time = (process_end - process_begin) / datetime.timedelta(microseconds=1)
        return {
                'result': {
                    'bucket': output_bucket,
                    'key': filename
                },
                'measurement': {
                    'download_time': download_time,
                    'download_size': download_size,
                    'upload_time': upload_time,
                    'upload_size': upload_size,
                    'compute_time': process_time
                }
            }