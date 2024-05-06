
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
import os

from torchvision.io import read_image
import torch
import torchvision
from torchvision import transforms
from torchvision import ops
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

DEFAULT_IMG_PATH = "data/images"
DEFAULT_ANNOTATION_PATH = "data/annotation_txts"

def parse_path(img_path, annotation_path):
    # List all files in the given directory
    file_names = os.listdir(img_path)
    
    # Filter out non-image files and remove extensions, assuming images end with typical image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    path_images = [f"{img_path}/{file_name}" for file_name in file_names]
    image_filenames_no_ext = [os.path.splitext(file)[0] for file in file_names if os.path.splitext(file)[1].lower() in image_extensions]
    assert len(image_filenames_no_ext) == len(path_images)
    path_annotations = [f"{annotation_path}/{file_name}.txt" for file_name in image_filenames_no_ext]
    return path_images, path_annotations

def parse_annotation(annotation_path):
    boxes = []
    labels = []
    with open(annotation_path, 'r') as file:
            for line in file:
                data = line.strip().split()
                class_label = int(data[0])
                class_label = 1 # the class in the text file is wrong

                # Convert coordinates to [x_min, y_min, x_max, y_max]
                x_min = float(data[1])
                y_min = float(data[2])
                x_max = float(data[3])
                y_max = float(data[4])
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_label)
    return torch.as_tensor(boxes, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.int64)

class BoarDataset(Dataset):
    '''
    A Pytorch Dataset class to load the images and their corresponding annotations.
    
    Returns
    ------------
    images: torch.Tensor of size (B, C, H, W)
    gt bboxes: torch.Tensor of size (B, max_objects, 4)
    gt classes: torch.Tensor of size (B, max_objects)
    '''
    def __init__(self, img_path=DEFAULT_IMG_PATH, annotation_path=DEFAULT_ANNOTATION_PATH, transform=None):
        self.annotation_path = annotation_path
        self.img_path = img_path
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(), 
        ])
        self.imgs, self.boxes, self.labels = self.get_data()
        
    def __len__(self):
        return self.imgs.size(dim=0)
    
    def __getitem__(self, idx):
        target = {}
        img = self.imgs[idx]
        target["boxes"] = torch.as_tensor(self.boxes[idx], dtype=torch.float32)
        target["labels"] = torch.as_tensor(self.labels[idx], dtype=torch.int64)
        return img, target
        
    def get_data(self):
        imgs = []
        classes = []
        boxes = []
        path_images, path_annotations = parse_path(img_path=self.img_path, annotation_path=self.annotation_path)
        for img_path, annotation_path in zip(path_images, path_annotations):

            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            boxes_tensor, labels_tensor = parse_annotation(annotation_path)
            imgs.append(img)
            boxes.append(boxes_tensor)
            classes.append(labels_tensor)
        
        imgs = torch.stack(imgs, dim=0)
        
        return imgs, boxes, classes