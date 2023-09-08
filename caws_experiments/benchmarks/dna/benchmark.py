def dna_visualization(download_path, *, _caws_output_dir):
    # using https://squiggle.readthedocs.io/en/latest/
    from squiggle import transform
    import uuid
    import datetime
    import os
    os.makedirs(_caws_output_dir, exist_ok=True)

    data = open(download_path, "r").read()

    process_begin = datetime.datetime.now()
    result = transform(data)
    process_end = datetime.datetime.now()

    output_path = os.path.join(_caws_output_dir, "visualization.json")
    with open(output_path, "w") as fp:
        json.dump(result, fp)

    process_time = (process_end - process_begin) / datetime.timedelta(microseconds=1)

    # TODO: Figure out output upload
    return {
            'result': {
                'path': output_path
            },
            'measurement': {
                'compute_time': process_time
            }
    }