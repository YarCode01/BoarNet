import os
import torch
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
from datetime import datetime
import albumentations as A
import numpy as np

def get_current_time():
    return datetime.now().strftime("%d-%m-%Y %H-%M-%S")

def get_augmentations():
    albumentation_list = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
    ]
    augmentations = A.Compose(
        albumentation_list,
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], check_each_transform=True)
    )
    return augmentations

def show_image_with_boxes(img_tensor, boxes_tensor, ax=None, save_path=None):
    """
    Display an image with bounding boxes.
    
    Parameters:
    - img_tensor: Tensor of the image data with pixel values in [0, 1].
    - boxes_tensor: Tensor of bounding boxes, each defined as [x_min, y_min, x_max, y_max].
    - ax: Matplotlib Axes object for plotting. If None, creates a new figure.
    - save_path: Path to save the plot image. If None, the plot won't be saved.
    """
    if ax is None:
        fig, ax = plt.subplots(1)
    
    img = F.to_pil_image(img_tensor)

    # Display the image
    ax.imshow(img)

    # Add the bounding boxes
    for box in boxes_tensor:
        x_min, y_min, x_max, y_max = box
        # Create a Rectangle patch
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    
    # Remove axis details
    ax.axis('off')

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


from sklearn.metrics import f1_score, precision_score, recall_score



def evaluate_box_similarity(targets, predictions, yolo_format_targets, yolo_format_predictions, threshold):
    """
    Evaluate the similarity between two arrays of bounding boxes.
    
    Parameters:
    - box1_array: Array of bounding boxes for the first set.
    - box2_array: Array of bounding boxes for the second set.
    - yolo_format: Whether the bounding boxes are in YOLO format. Default is True.
    
    Returns:
    - metrics: Dictionary containing evaluation metrics (IoU score, F1-score, precision, recall).
    """
    iou_scores = []
    img_dim = (300, 300)
    # Convert YOLO format to [x_min, y_min, x_max, y_max] if required
    if yolo_format_targets:
        for i, box in enumerate(targets):
            x_center = float(box[0]) * img_dim[0]
            y_center = float(box[1]) * img_dim[1]
            width = float(box[2]) * img_dim[0]
            height = float(box[3]) * img_dim[1]
            x_min = x_center - (width / 2)
            y_min = y_center - (height / 2)
            x_max = x_center + (width / 2)
            y_max = y_center + (height / 2)
            targets[i] = [x_min, y_min, x_max, y_max]
    
    if yolo_format_predictions:
        for i, box in enumerate(predictions):
            x_center = float(box[0]) * img_dim[0]
            y_center = float(box[1]) * img_dim[1]
            width = float(box[2]) * img_dim[0]
            height = float(box[3]) * img_dim[1]
            x_min = x_center - (width / 2)
            y_min = y_center - (height / 2)
            x_max = x_center + (width / 2)
            y_max = y_center + (height / 2)
            predictions[i] = [x_min, y_min, x_max, y_max]
    
     # Calculate IoU score for each pair of corresponding boxes
    detected = torch.zeros(len(targets))
    for target_box in targets:
        pass
        
    # Calculate F1-score, precision, and recall
    y_true = [int(iou >= threshold) for iou in iou_scores]
    y_pred = [1] * len(iou_scores)  # Assume all pairs are positive
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    metrics = {
        'IoU_score': sum(iou_scores) / len(iou_scores),
        'F1-score': f1,
        'Precision': precision,
        'Recall': recall
    }
    
    return metrics

def IoU_score_two_boxes(box1, box2):
    """
    Calculate Intersection over Union (IoU) score between two bounding boxes.
    
    Parameters:
    - box1: First bounding box, defined as [x_min, y_min, x_max, y_max].
    - box2: Second bounding box, defined as [x_min, y_min, x_max, y_max].
    
    Returns:
    - iou: Intersection over Union (IoU) score.
    """
    # Calculate intersection coordinates
    x_min_i = max(box1[0], box2[0])
    y_min_i = max(box1[1], box2[1])
    x_max_i = min(box1[2], box2[2])
    y_max_i = min(box1[3], box2[3])
    
    # Calculate intersection area
    inter_area = max(0, x_max_i - x_min_i) * max(0, y_max_i - y_min_i)
    
    # Calculate union area
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area_box1 + area_box2 - inter_area
    
    # Calculate IoU score
    iou = inter_area / union_area if union_area > 0 else 0.0
    
    return iou

def intersection_over_union(box, other_boxes):
    return torch.tensor([IoU_score_two_boxes(box, other_box) for other_box in other_boxes], dtype=torch.float32)

def nms(bboxes, confidence_scores, confidence_threshold, iou_threshold):
    """
    Performs Non Max Suppression
    
    Parameters:
        bboxes: Tensor containing a batch of bounding boxes
            type:tensor 
            shape:[N,4]
            format:[x1,y1,x2,y2]
        confidence_scores: Confidence scores for each bounding boxes
            type:tensor
            shape: 1D tensor of size N
        confidence_threshold: Minimum Confidence threshold required to keep the bounding box
            type:int
        iou_threshold: Overlapping Bounding boxes above this IoU threshold will be discarded.
            type:int
            
    Result:
        
    """
    bboxes = bboxes[confidence_scores.argsort(descending=True)]
    bboxes = bboxes[confidence_scores>confidence_threshold]
    chosen_bboxes = []
    while len(bboxes)>=2:
        chosen_bboxes.append(bboxes[0])
        other_bboxes = bboxes[1:]
        ious = intersection_over_union(bboxes[0], other_bboxes)
        bboxes = other_bboxes[ious<=iou_threshold]
    try:
        chosen_bboxes = torch.stack(chosen_bboxes)
    except:
        chosen_bboxes = torch.tensor([[]])
    return chosen_bboxes

def calculate_true_positives(predictions, scores, targets, yolo_format_targets, yolo_format_predictions, IoU_threshold):
    img_dim = (300, 300)
    if yolo_format_targets:
        for i, box in enumerate(targets):
            x_center = float(box[0]) * img_dim[0]
            y_center = float(box[1]) * img_dim[1]
            width = float(box[2]) * img_dim[0]
            height = float(box[3]) * img_dim[1]
            x_min = x_center - (width / 2)
            y_min = y_center - (height / 2)
            x_max = x_center + (width / 2)
            y_max = y_center + (height / 2)
            targets[i] = [x_min, y_min, x_max, y_max]
    
    if yolo_format_predictions:
        for i, box in enumerate(predictions):
            x_center = float(box[0]) * img_dim[0]
            y_center = float(box[1]) * img_dim[1]
            width = float(box[2]) * img_dim[0]
            height = float(box[3]) * img_dim[1]
            x_min = x_center - (width / 2)
            y_min = y_center - (height / 2)
            x_max = x_center + (width / 2)
            y_max = y_center + (height / 2)
            predictions[i] = [x_min, y_min, x_max, y_max]

    predictions = predictions.numpy()
    scores = scores.numpy()
    targets = targets.numpy()

    i = np.argsort(-scores)

    predictions = predictions[i]
    scores = scores[i]
    TP, FP = np.zeros(len(predictions)), np.zeros(len(predictions))
    detected = np.zeros(len(targets))

    for i, pred in enumerate(predictions):
        iouMax = float(0)
        for j in range(len(targets)):
            # print('Ground truth gt => %s' % (gt[j][3],))
            iou = IoU_score_two_boxes(pred, targets[j])
            if iou > iouMax:
                iouMax = iou
                jmax = j
        if iouMax >= IoU_threshold:
            if detected[jmax] == 0:
                TP[i] = 1  # count as true positive
                detected[jmax] = 1  # flag as already 'seen'
                # print("TP")
            else:
                FP[i] = 1  # count as false positive
                # print("FP")
        else:
            FP[i] = 1  # count as false positive
        




