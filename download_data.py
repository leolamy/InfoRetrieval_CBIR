import gdown
import zipfile
import os

DATASET_DIR = "./mini_dataset"
FILE_ID = "1Cc_E4zFEKZaZl9hq8VW-X89iYgZq99sD"
LOCAL_ZIP = "mini_dataset.zip"

def download_dataset():
    # Cas 1 : dataset déjà extrait
    if os.path.exists(DATASET_DIR):
        print("Dataset already present, skipping.")
        return

    if os.path.exists(LOCAL_ZIP):
        print(f"Found local {LOCAL_ZIP}, extracting...")
        with zipfile.ZipFile(LOCAL_ZIP, 'r') as z:
            z.extractall(".")
        return