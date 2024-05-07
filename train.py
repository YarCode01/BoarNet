from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, ssd300_vgg16, ssdlite320_mobilenet_v3_large
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch
import torchvision
#### Preparing the dataset
from dataset import BoarDataset


# Use the custom collate function in your DataLoader
def collate_fn(batch):
    images = [item['image'] for item in batch]
    targets = [item['target'] for item in batch]
    return images, targets

train_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to Tensor and scale to [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalization for pre-trained models
                         std=[0.229, 0.224, 0.225])
])
train_dataset = BoarDataset(transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
train_dataset[0]
for image, target in train_loader:
    print(target)
    break
model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=2, weights_backbone=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,trainable_backbone_layers=0)
device = None#torch.device("mps")
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        # images = [train_dataset[0]["image"]]
        # targets = [train_dataset[0]["target"]]
        images = [image.to(device) for image in images]
        targets = [{k : v.to(device) for k, v in target.items()} for target in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        print(loss_dict)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        # model.eval()
        # print(targets, model(images))
        model.train()
    
    

