import os

from tqdm import tqdm

from skimage.io import imsave

from pipeline import get_test_pipeline

from utils import read_image
from recognition import TEMPLATES_PATH

# BEGIN YOUR IMPORTS

# END YOUR IMPORTS

IMAGES_PATH = os.path.join(".", "sudoku_puzzles", "train")

# BEGIN YOUR CODE

"""
create dict of cell coordinates like in this example

CELL_COORDINATES = {"image_0.jpg": {'1': (0, 0),
                                    '2': (1, 1)},
                    "image_2.jpg": {'1': (2, 3),
                                    '3': (2, 1),
                                    '9': (5, 6)}}
"""

CELL_COORDINATES = # YOUR CODE

# END YOUR CODE

def main():
    os.makedirs(TEMPLATES_PATH, exist_ok=True)
    
    pipeline = get_test_pipeline()

    for file_name, coordinates_dict in CELL_COORDINATES.items():
        image_path = os.path.join(IMAGES_PATH, file_name)
        sudoku_image = read_image(image_path=image_path)
    
        # BEGIN YOUR CODE

        frontalized_image = # YOUR CODE
        sudoku_cells = # YOUR CODE
        
        # END YOUR CODE

        for digit, coordinates in tqdm(coordinates_dict.items(), desc=file_name):
            digit_templates_path = os.path.join(TEMPLATES_PATH, digit)
            os.makedirs(digit_templates_path, exist_ok=True)
            
            digit_template_path = os.path.join(digit_templates_path, f"{os.path.splitext(file_name)[0]}_{digit}.jpg")
            imsave(digit_template_path, sudoku_cells[*coordinates])


if __name__ == "__main__":
    main()
