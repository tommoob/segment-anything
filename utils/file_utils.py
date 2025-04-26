from pathlib import Path
import cv2

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}


def is_image_by_extension(image_path: str):
    return Path(image_path).suffix.lower() in IMAGE_EXTENSIONS


def just_real_image_paths(potential_image_list: list, image_path_root: str) -> list:
    """Returns a list of paths with an image extension.

    Parameters
    ----------
    potential_image_list
        list of potential images to check 
    image_path_root
        Root dir of all images
    """
    return [
        Path(image_path_root) / potential_image for potential_image in potential_image_list
        if is_image_by_extension(potential_image)
    ]


def load_image(image_path: str):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb