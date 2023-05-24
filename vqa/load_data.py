import os
import wget
import zipfile

# set the download directory and URLs for the dataset
download_dir = "./vqa_v2_abstract_scenes"
train_url = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Train_abstract_v002.zip"
val_url = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Val_abstract_v002.zip"
train_annotations_url = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Annotations_Train_abstract_v002.zip"
val_annotations_url = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Annotations_Val_abstract_v002.zip"

# create the download directory if it doesn't exist
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# download the training set questions
train_zip = os.path.join(download_dir, "train_questions.zip")
wget.download(train_url, train_zip)

# download the validation set questions
val_zip = os.path.join(download_dir, "val_questions.zip")
wget.download(val_url, val_zip)

# download the training set annotations
train_annotations_zip = os.path.join(download_dir, "train_annotations.zip")
wget.download(train_annotations_url, train_annotations_zip)

# download the validation set annotations
val_annotations_zip = os.path.join(download_dir, "val_annotations.zip")
wget.download(val_annotations_url, val_annotations_zip)

# extract the zip files
with zipfile.ZipFile(train_zip, "r") as zip_ref:
    zip_ref.extractall(download_dir)

with zipfile.ZipFile(val_zip, "r") as zip_ref:
    zip_ref.extractall(download_dir)

with zipfile.ZipFile(train_annotations_zip, "r") as zip_ref:
    zip_ref.extractall(download_dir)

with zipfile.ZipFile(val_annotations_zip, "r") as zip_ref:
    zip_ref.extractall(download_dir)

# delete the zip files
os.remove(train_zip)
os.remove(val_zip)
os.remove(train_annotations_zip)
os.remove(val_annotations_zip)



import os
import wget
import zipfile

# set the download directory and URLs for the dataset
train_download_dir = "./vqa_v2_abstract_scenes/AbstractScenes_v002_train2015"
val_download_dir = "./vqa_v2_abstract_scenes/AbstractScenes_v002_val2015"
train_url = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/scene_img/scene_img_abstract_v002_train2015.zip"
val_url = "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/scene_img/scene_img_abstract_v002_val2015.zip"

# create the download directory if it doesn't exist
if not os.path.exists(train_download_dir):
    os.makedirs(train_download_dir)
if not os.path.exists(val_download_dir):
    os.makedirs(val_download_dir)

# download the training set images
train_zip = os.path.join(train_download_dir, "train_images.zip")
wget.download(train_url, train_zip)

# download the validation set images
val_zip = os.path.join(val_download_dir, "val_images.zip")
wget.download(val_url, val_zip)

# extract the zip files
with zipfile.ZipFile(train_zip, "r") as zip_ref:
    zip_ref.extractall(train_download_dir)

with zipfile.ZipFile(val_zip, "r") as zip_ref:
    zip_ref.extractall(val_download_dir)

# delete the zip files
os.remove(train_zip)
os.remove(val_zip)