import warnings
warnings.filterwarnings("ignore")

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import numpy as np
from PIL import Image
import csv

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
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # apply transformation to image
    img = data_transforms(img)
    img = torch.unsqueeze(img, dim=0)
    return img

def processed_list(dir):
    original_list = getImages(dir)
    processed_list = []

    for img in original_list:
        processed_list.append(process_image(img))

    return processed_list

def classify_images(model, processed_img_list):
    classifications = []

    # evaluate model on image set
    for img in processed_img_list:
        model.eval()
        logits = model(img)
        classifications.append(torch.argmax(logits, dim=1).item())

    return classifications

def from_label(label):
    breed_labels = {}

    # get record of breeds
    with open('./records/breeds_data.csv', mode='r') as breeds:
        data = csv.reader(breeds)
        breed_labels = {rows[0]:rows[1] for rows in data}

    return breed_labels.get(label)


def main():
    # get model
    model_path = './models/resnet50_model.pt'
    model = torch.load(model_path)

    # get dog images
    image_directory = './my_dogs'
    img_list = processed_list(image_directory)

    # perform classification
    classification = classify_images(model, img_list)
    for label in classification:
        result = from_label(str(label))
        result = result.capitalize().replace('_', ' ')
        print(result)

if __name__ == "__main__":
    main()