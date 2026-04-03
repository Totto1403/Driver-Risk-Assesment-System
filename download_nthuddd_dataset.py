import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download("banudeep/nthuddd2")

# Target folder
target = "dataset/nthu"

# Create dataset folder if not exists
os.makedirs("dataset", exist_ok=True)

# Move dataset to your project folder
if not os.path.exists(target):
    shutil.move(path, target)

print("Dataset stored at:", target)