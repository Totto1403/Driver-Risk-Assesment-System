import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download("pankaj4321/fer-2013-facial-expression-dataset")

print("Downloaded to:", path)

# Destination inside your project
dest = "dataset/fer2013_val"

# Create folder if not exists
os.makedirs(dest, exist_ok=True)

# Copy files
for file in os.listdir(path):
    src_path = os.path.join(path, file)
    dst_path = os.path.join(dest, file)

    if os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    else:
        shutil.copy2(src_path, dst_path)

print("✅ Dataset copied to:", dest)