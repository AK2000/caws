def dna_visualization(download_path):
    # using https://squiggle.readthedocs.io/en/latest/
    from squiggle import transform
    data = open(download_path, "r").read()
    result = transform(data)

    buf = io.BytesIO(json.dumps(result).encode())

    # TODO: Figure out output upload
    return buf