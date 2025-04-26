""" This module instanciates SAM to remove the background (and foreground) of images containing cows"""
import click
import cv2
import numpy as np
import os
from pathlib import Path

from utils.file_utils import just_real_image_paths, load_image
from utils.model_utils import load_model
from segment_anything import SamPredictor
from workflow.where_are_the_bars import where_are_the_bars



def remove_cow_backgrounds(image_path: str, model_path: str):
    """Remove the backgrounds of cows using SAM model and save the resulting images.

    image_path
        The path to the image_files
    model_path
        The relative path to the SAM model weights

    """
    image_removal_locations = where_are_the_bars(image_path)

    potential_image_list = os.listdir(image_path)
    image_path_list = just_real_image_paths(potential_image_list, image_path)

    sam_predictor = load_model(model_path)

    for image_path in image_path_list:
        remove_image_background(image_path, sam_predictor, image_removal_locations[image_path])


def remove_image_background(
    image_path: str,
    sam_predictor: SamPredictor,
    input_point: dict[str, list],
    input_label: np.array = np.array([1]),
    ) -> None:

    image = load_image(image_path)

    sam_predictor.set_image(image)

    masks, scores, _ = sam_predictor.predict(
        point_coords=np.array(input_point),
        point_labels=input_label,
        multimask_output=True,
    )

    best_mask = masks[np.argmax(scores)]

    image_no_bar = image.copy()
    image_no_bar[best_mask] = [255, 255, 255]

    cv2.imshow("bar removed", image_no_bar)
    cv2.waitKey(4000)




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
@click.option(
    "--model_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path
    ),
    default="models/sam_vit_b_01ec64.pth"
)
def remove_cow_background_click(image_path: Path, model_path: str):
    remove_cow_backgrounds(image_path, model_path)


if __name__ == "__main__":
    remove_cow_background_click()
