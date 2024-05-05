
import numpy as np
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
from utils import *
from model import *
import os

import torch
import torchvision
from torchvision import ops
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

def parse(self):
    # List all files in the given directory
    file_names = os.listdir(self.img_path)
    
    # Filter out non-image files and remove extensions, assuming images end with typical image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    path_images = [f"{self.img_path}/{file_name}" for file_name in file_names]
    image_filenames_no_ext = [os.path.splitext(file)[0] for file in file_names if os.path.splitext(file)[1].lower() in image_extensions]
    path_annotations = [f"{self.img_path}/{file_name}.txt" for file_name in image_filenames_no_ext]
    return path_images, path_annotations

class DetectionDataset(Dataset):
    '''
    A Pytorch Dataset class to load the images and their corresponding annotations.
    
    Returns
    ------------
    images: torch.Tensor of size (B, C, H, W)
    gt bboxes: torch.Tensor of size (B, max_objects, 4)
    gt classes: torch.Tensor of size (B, max_objects)
    '''
    def __init__(self, img_path, annotation_path):
        self.annotation_path = annotation_path
        self.img_path = img_path
        
        self.img_data_all, self.gt_bboxes_all, self.gt_classes_all = self.get_data()
        
    def __len__(self):
        return self.img_data_all.size(dim=0)
    
    def __getitem__(self, idx):
        return self.img_data_all[idx], self.gt_bboxes_all[idx], self.gt_classes_all[idx]
        
    def get_data(self):
        img_data_all = []
        gt_idxs_all = []
        
        file_names = parse(self.annotation_path, self.img_dir)
        
        for i, file_name in enumerate(file_names):
            img_path
            # skip if the image path is not valid
            if not os.path.exists(img_path):
                continue
                
            # read and resize image
            img = io.imread(img_path)
            
            # convert image to torch tensor and reshape it so channels come first
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            
            # encode class names as integers
            gt_classes = gt_classes_all[i]
            gt_idx = torch.Tensor([self.name2idx[name] for name in gt_classes])
            
            img_data_all.append(img_tensor)
            gt_idxs_all.append(gt_idx)
        
        # pad bounding boxes and classes so they are of the same size
        gt_bboxes_pad = pad_sequence(gt_boxes_all, batch_first=True, padding_value=-1)
        gt_classes_pad = pad_sequence(gt_idxs_all, batch_first=True, padding_value=-1)
        
        # stack all images
        img_data_stacked = torch.stack(img_data_all, dim=0)
        
        return img_data_stacked.to(dtype=torch.float32), gt_bboxes_pad, gt_classes_pad