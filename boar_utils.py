import os
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
from datetime import datetime
import albumentations as A

def get_current_time():
    return datetime.now().strftime("%d-%m-%Y %H-%M-%S")

def get_augmentations():
    albumentation_list = [
        # A.RandomCrop(p=1, height=220, width=220),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.Resize(height=512, width=512, p=1),
            # A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
    ]
    augmentations = A.Compose(
        albumentation_list,
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
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

def create_gif(image_folder, gif_path, duration=0.5):
    """
    Create a GIF from a series of JPEG images in a folder.

    Parameters:
    - image_folder: Path to the folder containing JPEG images.
    - gif_path: Path to save the GIF file.
    - duration: Duration of each frame in seconds (default is 0.5 seconds).
    """
    images = []
    # Iterate through all the JPEG files in the folder
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            file_path = os.path.join(image_folder, filename)
            images.append(imageio.imread(file_path))

    # Save the images as a GIF
    imageio.mimsave(gif_path, images, duration=duration)

from sklearn.metrics import f1_score, precision_score, recall_score

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
    for box1, box2 in zip(targets, predictions):
        iou = IoU_score_two_boxes(box1, box2)
        iou_scores.append(iou)
        
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

