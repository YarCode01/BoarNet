from torchvision.models.detection import ssdlite320_mobilenet_v3_large,fasterrcnn_mobilenet_v3_large_fpn, ssd300_vgg16
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch
import torchvision
from tqdm import tqdm
import torchvision.transforms.functional as F
from dataset import BoarDataset
import os
import json
from boar_utils import get_current_time, get_augmentations
from  torchvision.transforms import v2
from albumentations.pytorch import ToTensorV2
import albumentations as A

CURRENT_TIME = get_current_time()
MODEL_PATH = f"models/{CURRENT_TIME}"
RESULTS_PATH = f"results/losses/{CURRENT_TIME}"
MODEL_NAME = "fasterrcnn_mobilenet_v3_large_fpn"
os.mkdir(MODEL_PATH)
os.mkdir(RESULTS_PATH)

TRAIN_IMG_PATH = "data/train/images"
TRAIN_ANNOTATION_PATH = "data/train/labels"

TEST_IMG_PATH = "data/test/images"
TEST_ANNOTATION_PATH = "data/test/labels"

# Use the custom collate function in your DataLoader
def collate_fn(batch):
    images = [item['image'] for item in batch]
    targets = [item['target'] for item in batch]
    return images, targets

train_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalization is not needed as it is done by pytorch by default
                         std=[0.229, 0.224, 0.225])
])

augmentations = get_augmentations()

train_dataset = BoarDataset(img_path=TRAIN_IMG_PATH, annotation_path=TRAIN_ANNOTATION_PATH,transform=train_transform, augmentations=augmentations, yolo_format=True)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn,drop_last=True)

test_dataset = BoarDataset(img_path=TEST_IMG_PATH, annotation_path=TEST_ANNOTATION_PATH,transform=train_transform, yolo_format=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, drop_last=True)

# model = ssdlite320_mobilenet_v3_large(num_classes=2, weights_backbone=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1, trainable_backbone_layers=5)
model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=2, weights_backbone=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT, trainable_backbone_layers=0)
device = None#torch.device("mps")
model = model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
num_epochs = 30

train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    #training
    for images, targets in tqdm(train_loader):
        images = [image.to(device) for image in images]
        targets = [{k : v.to(device) for k, v in target.items()} for target in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_train_loss += losses.item()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    #evaluation 
    epoch_test_loss = 0
    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_test_loss += losses.item()

    test_losses.append(epoch_test_loss / len(test_loader))
    train_losses.append(epoch_train_loss / len(train_loader))

    print("EPOCH: ", epoch)
    print("TEST LOSS: ", test_losses[-1])
    print("TRAIN LOSS: ", train_losses[-1])
    torch.save(model.state_dict(), f"{MODEL_PATH}/epoch_{epoch}_{MODEL_NAME}.pth")
    with open(f'{RESULTS_PATH}/train_losses.json', 'w') as f:
        json.dump(train_losses, f, indent=4)

    with open(f'{RESULTS_PATH}/test_losses.json', 'w') as f:
        json.dump(test_losses, f, indent=4)


