""" This module instanciates SAM to remove the background (and foreground) of images containing cows"""
import click
from collections import defaultdict
import cv2
import os
from pathlib import Path

from utils.file_utils import just_real_image_paths, load_image


def where_are_the_bars(image_path_root: str):
    """Remove the backgrounds of cows using SAM model and save the resulting images.

    image_path
        The path to the image_files
    model_path
        The relative path to the SAM model weights

    """
    total_click_locations = defaultdict(list)
    potential_image_list = os.listdir(image_path_root)
    image_path_list = just_real_image_paths(potential_image_list, image_path_root)

    for image_path in image_path_list:
        coordinates_of_click = get_location_object_to_remove(image_path)
        total_click_locations[image_path] = coordinates_of_click

    return total_click_locations


def get_location_object_to_remove(image_path: str, wait_time_seconds: float = 3) -> list:
    coordinates_of_click = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinates_of_click.append((x, y))
            cv2.destroyAllWindows()
    image = load_image(image_path)

    cv2.imshow(f"Click on the bar. Image {image_path}", image)

    cv2.setMouseCallback(f"Click on the bar. Image {image_path}", click_event)

    cv2.waitKey(wait_time_seconds * 1000)
    cv2.destroyAllWindows()

    return coordinates_of_click if coordinates_of_click else []

@click.command()
@click.option(
    "--image_path",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path
    )
)
def where_are_the_bars_click(image_path: Path):
    where_are_the_bars(image_path)


if __name__ == "__main__":
    where_are_the_bars_click()
