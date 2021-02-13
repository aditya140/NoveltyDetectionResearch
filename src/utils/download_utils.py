import os, requests
import tqdm


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""

    def process_response(r):
        chunk_size = 16 * 1024
        total_size = int(r.headers.get("Content-length", 0))
        with open(path, "wb") as file:
            with tqdm(
                total=total_size, unit="B", unit_scale=1, desc=path.split("/")[-1]
            ) as t:
                for chunk in r.iter_content(chunk_size):
                    if chunk:
                        file.write(chunk)
                        t.update(len(chunk))

    if "drive.google.com" not in url:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
        process_response(response)
        return

    print("downloading from Google Drive; may take a few minutes")
    id_ = url.split("/")[-2]
    print("downloading to ",path)
    download_file_from_google_drive(id_, destination=path)
    return