import gdown
import zipfile
from config import ARTIFACTS_DIR, ZIP_PATH, EXTRACT_DIR, FILE_ID

def download_and_extract_data():
    """Downloads and extracts the dataset."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    if not ZIP_PATH.exists():
        print("Downloading dataset...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, str(ZIP_PATH), quiet=False)
    else:
        print("Dataset already downloaded.")

    if not EXTRACT_DIR.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print(f"Extracted to {EXTRACT_DIR}")
    else:
        print("Dataset already extracted.")

if __name__ == "__main__":
    download_and_extract_data()