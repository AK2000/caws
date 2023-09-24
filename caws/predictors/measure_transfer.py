from urllib.parse import urlparse
import subprocess

from caws.transfer import TransferManager

def main(endpoint_id):
    manager = TransferManager()
    endpoint = manager.transfer_client.get_endpoint(endpoint_id)
    url = endpoint["DATA"][0]["uri"]
    parsed_url = urlparse(url).netloc.partition(":")[0]
    completed_process = subprocess.run(["tracepath", parsed_url], capture_output=True)
    nhops = completed_process.stdout.count(b"\n") - 1
    print("Number of hops: ", nhops)

if __name__ == "__main__":
    main("5bf3e9c5-c6d0-45d8-b63a-9e33a1891319")