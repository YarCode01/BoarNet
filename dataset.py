from PIL import Image
import matplotlib.patches as patches
import os
import cv2
from torchvision.io import read_image
import torch
from torchvision import transforms
from torch.utils.data import  Dataset
from torchvision import tv_tensors

DEFAULT_IMG_PATH = "data/train/images"
DEFAULT_ANNOTATION_PATH = "data/train/labels"

def parse_path(img_path, annotation_path):
    # List all files in the given directory
    file_names = os.listdir(img_path)
    
    # Filter out non-image files and remove extensions, assuming images end with typical image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    path_images = [f"{img_path}/{file_name}" for file_name in file_names if ".jpg" in file_name]
    image_filenames_no_ext = [os.path.splitext(file)[0] for file in file_names if os.path.splitext(file)[1].lower() in image_extensions]
    assert len(image_filenames_no_ext) == (len(path_images))
    path_annotations = [f"{annotation_path}/{file_name}.txt" for file_name in image_filenames_no_ext]
    return path_images, path_annotations

def parse_annotation(annotation_path, img_dim = None, yolo_format=False):
    boxes = []
    labels = []
    with open(annotation_path, 'r') as file:
            for line in file:
                data = line.strip().split()
                # class_label = int(data[0])
                class_label = 0

                # Convert coordinates to [x_min, y_min, x_max, y_max]
                if yolo_format:
                # Convert YOLO format to [x_min, y_min, x_max, y_max]
                    x_center = float(data[1]) * img_dim[0]
                    y_center = float(data[2]) * img_dim[1]
                    width = float(data[3]) * img_dim[0]
                    height = float(data[4]) * img_dim[1]
                    x_min = x_center - (width / 2)
                    y_min = y_center - (height / 2)
                    x_max = x_center + (width / 2)
                    y_max = y_center + (height / 2)
                else:
                    # Assuming data is already in [x_min, y_min, x_max, y_max] format
                    x_min = float(data[1])
                    y_min = float(data[2])
                    x_max = float(data[3])
                    y_max = float(data[4])

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_label)
    return boxes, labels


class BoarDataset(Dataset):
    def __init__(self, img_path=DEFAULT_IMG_PATH, annotation_path=DEFAULT_ANNOTATION_PATH, transform=None, yolo_format=False, augmentations=None):
        self.annotation_path = annotation_path
        self.img_path = img_path
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(), 
        ])
        self.augmentations = augmentations
        self.yolo_format = yolo_format
        self.imgs, self.boxes, self.labels = self.get_data()
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        target = {}
        bboxes = self.boxes[idx]
        labels = self.labels[idx]
        
        if self.augmentations:
            transformed = self.augmentations(image=img, bboxes=bboxes, labels=labels)
            img = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']

        bboxes_tensor = torch.Tensor(bboxes)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        img = self.transform(img)
        target["boxes"] = bboxes_tensor
        target["labels"] = labels_tensor
        return {"image": img, "target": target}
        
    def get_data(self):
        imgs = []
        classes = []
        boxes = []
        path_images, path_annotations = parse_path(img_path=self.img_path, annotation_path=self.annotation_path)
        for img_path, annotation_path in zip(path_images, path_annotations):
            
            img = cv2.imread(img_path)
            img_dim = img.shape[:2]
            boxes_array, labels_array = parse_annotation(annotation_path, img_dim, yolo_format=self.yolo_format)
            imgs.append(img)
            boxes.append(boxes_array)
            classes.append(labels_array)
        
        return imgs, boxes, classes