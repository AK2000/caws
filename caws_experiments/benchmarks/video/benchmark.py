from caws.task import caws_task
from caws.features import ArgFeature, ArgAttrFeature, CawsFeatureType

def video_processing(download_path, duration, op, *, _caws_output_dir, watermark_path=None):
    import datetime
    import os
    import stat
    import subprocess
    
    os.makedirs(_caws_output_dir, exist_ok=True)

    def call_ffmpeg(args):
        # TODO: Ensure ffmpeg is on path
        ret = subprocess.run(['ffmpeg', '-y'] + args, check=True)
        if ret.returncode != 0:
            print('Invocation of ffmpeg failed!')
            print('Out: ', ret.stdout.decode('utf-8'))
            raise RuntimeError()

    # https://superuser.com/questions/556029/how-do-i-convert-a-video-to-gif-using-ffmpeg-with-reasonable-quality
    def to_gif(video, duration):
        output = os.path.join(_caws_output_dir, 'processed-{}.gif'.format(os.path.basename(video)))
        call_ffmpeg(["-i", video,
            "-t",
            "{0}".format(duration),
            "-vf",
            "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop", "0",
            output])
        return output

    # https://devopstar.com/2019/01/28/serverless-watermark-using-aws-lambda-layers-ffmpeg/
    def watermark(video, duration):
        output = os.path.join(_caws_output_dir, 'processed-{}.gif'.format(os.path.basename(video)))
        call_ffmpeg([
            "-i", video,
            "-i", watermark_path,
            "-t", "{0}".format(duration),
            "-filter_complex", "overlay=main_w/2-overlay_w/2:main_h/2-overlay_h/2",
            output])
        return output

    def transcode_mp3(video, duration):
        pass

    operations = { 'transcode' : transcode_mp3, 'extract-gif' : to_gif, 'watermark' : watermark }

    process_begin = datetime.datetime.now()
    output_path = operations[op](download_path, duration)
    process_end = datetime.datetime.now()

    process_time = (process_end - process_begin) / datetime.timedelta(microseconds=1)
    return {
            'result': {
                'path': output_path
            },
            'measurement': {
                'compute_time': process_time
            }
        }
features = [ArgAttrFeature("size"), ArgFeature(arg_name="duration"), ArgFeature(arg_name="op", feature_type=CawsFeatureType.CATEGORICAL)]
video_processing = caws_task(video_processing, features=features)