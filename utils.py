import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
def show_image_with_boxes(img_tensor, boxes_tensor, ax=None):
    """
    Display an image with bounding boxes.
    
    Parameters:
    - img_tensor: Tensor of the image data with pixel values in [0, 1].
    - boxes_tensor: Tensor of bounding boxes, each defined as [x_min, y_min, x_max, y_max].
    - ax: Matplotlib Axes object for plotting. If None, creates a new figure.
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