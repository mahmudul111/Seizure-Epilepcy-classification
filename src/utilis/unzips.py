import os
import zipfile

def unzip_file(zip_path, extract_to):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"The zip file {zip_path} does not exist.")
    
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")


if __name__ == "__main__":
    zip_path = "model/saved_models.zip"
    extract_to = "extracted_models"
    unzip_file(zip_path, extract_to)