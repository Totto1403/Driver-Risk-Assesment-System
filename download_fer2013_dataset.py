import kagglehub
import shutil
import os

# Download dataset (goes to cache)
path = kagglehub.dataset_download("msambare/fer2013")

# Target directory
target_path = "dataset/fer2013"

# Move dataset to your desired folder
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Move entire folder
if not os.path.exists(target_path):
    shutil.move(path, target_path)

print("Dataset stored at:", target_path)