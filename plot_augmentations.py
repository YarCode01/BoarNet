# !matplotlib
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, ssd300_vgg16, ssdlite320_mobilenet_v3_large
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
from dataset import BoarDataset
from boar_utils import show_image_with_boxes, get_augmentations
import os
import json
from boar_utils import get_current_time
INDEX = 3
CURRENT_TIME = "10-05-2024 16-54-35"
# MODEL_PATH = f"models/{CURRENT_TIME}/epoch_4_fasterrcnn_mobilenet_v3_large_fpn.pth"
MODEL_PATH = f"epoch_24_mobilnet.pth"
model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=2, weights_backbone=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2, trainable_backbone_layers=0)
# model = ssdlite320_mobilenet_v3_large(num_classes=2, weights_backbone=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1, trainable_backbone_layers=0)
# model.load_state_dict(torch.load(MODEL_PATH))
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
TEST_IMG_PATH = "data/test/images"
TEST_ANNOTATION_PATH = "data/test/labels"
# TEST_IMG_PATH = "data/Wild Boar.v1i.yolov7pytorch/train/images"
# TEST_ANNOTATION_PATH = "data/Wild Boar.v1i.yolov7pytorch/labels"
model.eval()
train_transform = transforms.Compose([
    transforms.ToTensor(), 
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalization is not needed as it is done by pytorch by default
    #                      std=[0.229, 0.224, 0.225])
])
augmentations = get_augmentations()
test_dataset = BoarDataset(img_path=TEST_IMG_PATH, annotation_path=TEST_ANNOTATION_PATH,transform=train_transform, augmentations=augmentations, yolo_format=True)
test_dataset_vis = BoarDataset(img_path=TEST_IMG_PATH, annotation_path=TEST_ANNOTATION_PATH, augmentations=augmentations, yolo_format=True)
test_dataset_vis_orig = BoarDataset(img_path=TEST_IMG_PATH, annotation_path=TEST_ANNOTATION_PATH, yolo_format=True)
images = [test_dataset[INDEX]['image']]
# images = torch.tensor(images,dtype=torch.float32)
# images.shape
# boxes = model(images)
THRESHOLD = 0.4
# boxes_score_zip = zip(boxes[0]['boxes'], boxes[0]['scores'])
# boxes = [box.tolist() for box, score in boxes_score_zip if score > THRESHOLD]
# show_image_with_boxes(test_dataset_vis[INDEX]['image'], boxes)
sample = test_dataset_vis[INDEX]
show_image_with_boxes(sample['image'],sample['target']['boxes'].tolist())
show_image_with_boxes(test_dataset_vis_orig[INDEX]['image'],test_dataset_vis_orig[INDEX]['target']['boxes'].tolist())