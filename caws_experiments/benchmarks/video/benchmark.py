def video_processing(download_path, op):
    import datetime
    import os
    import stat
    import subprocess

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
    upload_path = operations[op](download_path, duration, event)

    #TODO: Figure out upload of output
    return upload_path