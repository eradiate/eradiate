import os
import sys
from os.path import basename
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile

import click
import requests
import ruamel.yaml as yaml
import tqdm

import eradiate

DOWNLOAD_DIR = eradiate.config.download_dir

with open(DOWNLOAD_DIR / "files.yml", "r") as f:
    DESTINATION = yaml.safe_load(f)


def url_to_name(url: str) -> str:
    return basename(urlsplit(url).path)


def download_url(url: str, chunk_size: int = 1024) -> Path:
    """
    Download from url and write result to a file.

    Parameters
    ----------
    url: str
        URL.

    chunk_size: int
        Chunk size in bytes.

    Returns
    -------
    Path
        Path to where the archive file is saved.
    """
    # Inspired from:
    # https://github.com/sirbowen78/lab/blob/master/file_handling/dl_file1.py

    filesize = int(requests.head(url).headers["Content-Length"])
    file_name = url_to_name(url=url)
    path = Path(eradiate.config.dir, file_name)

    with requests.get(url=url, stream=True) as r, open(path, "wb") as f, tqdm.tqdm(
        unit="B",  # unit string to be displayed.
        unit_scale=True,  # let tqdm to determine the scale in kilo, mega..etc.
        unit_divisor=1024,  # is used when unit_scale is true
        total=filesize,
        file=sys.stdout,  # default goes to stderr, this is the display on console.
        desc=file_name,
    ) as progress:
        for chunk in r.iter_content(chunk_size=chunk_size):
            datasize = f.write(chunk)
            progress.update(datasize)

    return path


def extract(file: Path, destination: Path) -> None:
    """
    Extract an archive file and deletes it.

    Parameters
    ----------
    file: Path
        Path to the archive file.

    file: Path
        Where to extract the archive file.
    """
    with ZipFile(file, "r") as zip:
        zip.extractall(destination)
    os.remove(file)


@click.command()
def cli():
    """
    Command-line interface to download additional data for the Eradiate
    radiative transfer model.
    """
    root_url = "https://eradiate.eu/data/"

    print(f"Downloading from {root_url}")

    for archive in tqdm.tqdm(DESTINATION, desc="Global progress", leave=True):
        url = root_url + archive
        archive_file = download_url(url=url)
        tqdm.tqdm.write(f"Extracting archive ...")
        extract(file=archive_file, destination=DOWNLOAD_DIR / DESTINATION[archive])

    print("Done")


if __name__ == "__main__":
    cli()
