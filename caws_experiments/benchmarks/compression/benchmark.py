def compression(directory_path, *, _caws_output_dir):
    import os
    import datetime
    import shutil
    
    os.makedirs(_caws_output_dir, exist_ok=True)

    def parse_directory(directory):
        size = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                size += os.path.getsize(os.path.join(root, file))
        return size

    size = parse_directory(directory_path)
    key = "compressed"
    compress_begin = datetime.datetime.now()
    shutil.make_archive(os.path.join(_caws_output_dir, key), 'zip', root_dir=directory_path)
    compress_end = datetime.datetime.now()

    archive_name = '{}.zip'.format(key)
    archive_path = os.path.join(_caws_output_dir, archive_name)
    archive_size = os.path.getsize(archive_path)
    process_time = (compress_end - compress_begin) / datetime.timedelta(microseconds=1)

    return {
            'result': {
                'path': archive_path
            },
            'measurement': {
                'download_size': size,
                'upload_size': archive_size,
                'compute_time': process_time
            }
        }