import os
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
from datetime import datetime

def get_current_time():
    return datetime.now().strftime("%d-%m-%Y %H-%M-%S")

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