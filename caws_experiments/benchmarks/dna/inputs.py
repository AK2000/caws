from caws.path import CawsPath

def generate_inputs(src_endpoint, size, data_dir=""):

    for file in glob.glob(os.path.join(data_dir, "500.scientific", "504.dna-visualisation", '*.fasta')):
        return (CawsPath(src_endpoint, file, isolate=True),), {}