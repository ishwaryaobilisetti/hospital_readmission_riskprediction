import os
import zipfile
import requests
from tqdm import tqdm

def download_dataset():
    url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    zip_path = os.path.join(data_dir, "dataset.zip")
    
    print(f"Downloading dataset from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    
    with open(zip_path, "wb") as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
            
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)
        
    print("Cleaning up temporary zip file...")
    os.remove(zip_path)
    
    print(f"Dataset successfully downloaded and extracted to: {data_dir}")
    print("Files in directory:")
    for f in os.listdir(data_dir):
        print(f" - {f}")

if __name__ == "__main__":
    download_dataset()
