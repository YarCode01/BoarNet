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
from boar_utils import show_image_with_boxes, nms
import os
import json
from boar_utils import get_current_time
INDEX = 5
CURRENT_TIME = "15-05-2024 16-55-34"
RESULTS_PATH = f"results/predictions/{CURRENT_TIME}/"
try:
    os.mkdir(RESULTS_PATH)
except:
    pass
MODEL_PATH = f"models/{CURRENT_TIME}/epoch_17_fasterrcnn_mobilenet_v3_large_fpn.pth"
model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=2, weights_backbone=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT, trainable_backbone_layers=0)
model.load_state_dict(torch.load(MODEL_PATH))
TEST_IMG_PATH = "data/test/images"
TEST_ANNOTATION_PATH = "data/test/labels"
model.eval()
train_transform = transforms.Compose([
    transforms.ToTensor(), 
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalization is not needed as it is done by pytorch by default
    #                      std=[0.229, 0.224, 0.225])
])
test_dataset = BoarDataset(img_path=TEST_IMG_PATH, annotation_path=TEST_ANNOTATION_PATH,transform=train_transform, yolo_format=True)
test_dataset_vis = BoarDataset(img_path=TEST_IMG_PATH, annotation_path=TEST_ANNOTATION_PATH, yolo_format=True)
for index in range(len(test_dataset)):
    print(index)
    images = [test_dataset[index]['image']]
    boxes = model(images)
    supressed_boxes = nms(bboxes=boxes[0]['boxes'], confidence_scores=boxes[0]['scores'],confidence_threshold=0.3, iou_threshold=0.3)
    try:
        show_image_with_boxes(test_dataset_vis[index]['image'], supressed_boxes['boxes'].tolist(), save_path=RESULTS_PATH+f"{index}.jpg")
    except:
        pass