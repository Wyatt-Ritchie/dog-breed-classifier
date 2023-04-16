import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# order of algorithm
# 1. get list of images
# 2. process all the images
# 3. classify the images and add the classifications to a list
# 4. print the list

"""
To finish this we need to create a directory that contains images of type .jpg
Also need to include a model in the directory, can't to over github I don't think
Include the model just below. Test
"""
model_path = './models/resnet50_model.pt'
model = torch.load(model_path)

image_directory = './my_dogs'

def getImages(directoryName):
    dir_path = directoryName
    file_pattern = '*.jpg'
    image_list = []
    for file_path in glob.glob(os.path.join(dir_path, file_pattern)):
        img = Image.open(file_path)
        img_arr = np.array(img)
        image_list.append(img_arr)
    return image_list

def process_image(img):
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize(256)])
    trans2 = transforms.Compose([transforms.CenterCrop(224)])
    trans3 = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_trans1 = trans(img)
    img_trans2 = trans2(img_trans1)
    img_trans3 = trans3(img_trans2)
    img_trans1 = np.transpose(img_trans1, (1, 2, 0))
    img_trans2 = np.transpose(img_trans2, (1, 2, 0))
    img_trans3 = np.transpose(img_trans3, (1, 2, 0))
    return img_trans3

def processed_list():
    original_list = getImages(image_directory)
    processed_list = []
    for img in original_list:
        processed_list.append(process_image(img))
    print(processed_list[0].shape)
    return processed_list

def classify_images(model, processed_img_list):
    classifications = []
    for img in processed_img_list:
        pred = model.forward(img)
        classifications.append(pred)
    return classifications

img_list = processed_list()
classification = classify_images(model, img_list)

print(classification)
