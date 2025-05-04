import gdown
import os

folder_url = "" # Left blank for security reasons, but this code maybe useful

# Extract the folder ID from the URL
folder_id = folder_url.split('/')[-1]
output_directory = "data/raw"
os.makedirs(output_directory, exist_ok=True)
full_gdrive_url = f'https://drive.google.com/uc?id={folder_id}&export=download'

print(f"Attempting to download files from Google Drive folder ID: {folder_id}")
print(f"Saving files to: {os.path.abspath(output_directory)}")

try:
    # Download the folder contents
    # Note: gdown might download a zip file if the folder is large, 
    # or individual files if it's small. This script assumes direct file downloads.
    # If it downloads a zip, you'll need to unzip it manually or add unzipping logic.
    gdown.download_folder(folder_url, output=output_directory, quiet=False, use_cookies=False)
    print("Download attempted. Please check the output directory for files.")
    print("If a zip file was downloaded, you may need to extract it.")

except Exception as e:
    print(f"An error occurred during download: {e}")
    print("Please ensure the Google Drive folder is public or shared correctly.")
    print("You might also need to install gdown: pip install gdown") 