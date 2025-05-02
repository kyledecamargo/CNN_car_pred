import os
import shutil
import scipy.io

# paths
base_dir = "data"
devkit_dir = os.path.join(base_dir, "car_devkit")
train_images_dir = os.path.join(base_dir, "cars_train")
output_train_dir = os.path.join(base_dir, "cars_train_organized", "train")

# load metadata and annotations
meta = scipy.io.loadmat(os.path.join(devkit_dir, "cars_meta.mat"))
annotations = scipy.io.loadmat(os.path.join(devkit_dir, "cars_train_annos.mat"))

# get car class names and annotations
class_names = [c[0] for c in meta["class_names"][0]]
annos = annotations["annotations"][0]

# create folders and organize images
for anno in annos:
    file_name = anno['fname'][0]
    class_id = anno['class'][0][0] - 1  # MATLAB is 1-indexed
    class_name = class_names[class_id]
    
    class_dir = os.path.join(output_train_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    src_path = os.path.join(train_images_dir, file_name)
    dst_path = os.path.join(class_dir, file_name)
    
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"Missing file: {src_path}")

print("✅ Dataset organized successfully into:", output_train_dir)