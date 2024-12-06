import os
import zipfile
import shutil
files = os.listdir()
files = [f for f in files if "zip" in f]
splits = ["train","valid"]


def create_folder(name):
    if not os.path.exists(name):
        os.mkdir(name)
    
create_folder("datasets")

for file in files:

    create_folder(file[:-3])

    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(f"{file[:-3]}/")
    if "boxes" in file:
        continue

    images = os.listdir(f"{file[:-3]}/")
    images = [i for i in images if "box" not in i]

    train_split = int(len(images)*0.9)
    create_folder(f"datasets/{file[:-3]}")
    create_folder(f"datasets/{file[:-3]}/train")
    create_folder(f"datasets/{file[:-3]}/train/images")
    create_folder(f"datasets/{file[:-3]}/train/labels")
    for image in images[:train_split]:
        label_name = image.split(".")[0]+".txt"
        shutil.copy(f"{file[:-3]}/{image}", f"datasets/{file[:-3]}/train/images/{image}")
        shutil.copy(f"boxes/{label_name}", f"datasets/{file[:-3]}/train/labels/{label_name}")

    create_folder(f"datasets/{file[:-3]}/valid")
    create_folder(f"datasets/{file[:-3]}/valid/images")
    create_folder(f"datasets/{file[:-3]}/valid/labels")
    for image in images[train_split:]:
        label_name = image.split(".")[0]+".txt"
        shutil.copy(f"{file[:-3]}/{image}", f"datasets/{file[:-3]}/valid/images/{image}")
        shutil.copy(f"boxes/{label_name}", f"datasets/{file[:-3]}/valid/labels/{label_name}")
    print(f"Created train and valid split for {file}")

outdoor_images = os.listdir("front")
train_split = int(len(outdoor_images)*0.9)
create_folder(f"datasets/outdoors")
create_folder(f"datasets/outdoors/train")
create_folder(f"datasets/outdoors/train/images")
create_folder(f"datasets/outdoors/train/labels")
for image in outdoor_images[:train_split]:
    label_name = image.split(".")[0]+"_bg1.txt"
    shutil.copy(f"front/{image}",f"datasets/outdoors/train/images/{image}")
    shutil.copy(f"boxes/{label_name}", f"datasets/outdoors/train/labels/{label_name.split('_')[0]}.txt")

create_folder(f"datasets/outdoors/valid")
create_folder(f"datasets/outdoors/valid/images")
create_folder(f"datasets/outdoors/valid/labels")
for image in outdoor_images[train_split:]:
    label_name = image.split(".")[0]+"_bg1.txt"
    shutil.copy(f"front/{image}",f"datasets/outdoors/valid/images/{image}")
    shutil.copy(f"boxes/{label_name}", f"datasets/outdoors/valid/labels/{label_name.split('_')[0]}.txt")
print(f"Created train and valid split for outdoors")


