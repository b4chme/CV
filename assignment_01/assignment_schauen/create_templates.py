import os

from tqdm import tqdm

from skimage.io import imsave

from pipeline import get_test_pipeline

from utils import read_image
from recognition import TEMPLATES_PATH

# BEGIN YOUR IMPORTS
from frontalization import frontalize_image
from recognition import get_sudoku_cells

# END YOUR IMPORTS

IMAGES_PATH = os.path.join(".", "sudoku_puzzles", "train")

# BEGIN YOUR CODE

"""
create dict of cell coordinates like in this example

CELL_COORDINATES = {
                    
                    "image_0.jpg": {'1': (0, 0),
                                    '2': (1, 1)},
                    "image_2.jpg": {'1': (2, 3),
                                    '3': (2, 1),
                                    '9': (5, 6)}
                    }
"""
74
CELL_COORDINATES = { 
    "image_0.jpg": {'8': (0, 5),
                    '9': (2, 0),
                    '3': (2, 1),
                    '5': (4, 0)
                    },

    "image_4.jpg": {
        '6': (0, 2),
        '2': (1, 0),
        '1': (3, 0),
        '7':( 8, 3)
    },

    "image_8.jpg": {
    '4': (0, 0),
    '2': (0, 8),
    '1': (1, 3),
    '9': (0, 6)
    },

    "image_5.jpg": {
    '5': (1, 1),
    '3': (1, 7),
    '6': (1, 4),
    '8': (8, 0)
    },

    "image_2.jpg": {
    '7': (2, 8), # '7': (8, 1),
    '4': (7, 5)
    }
    }

# END YOUR CODE

def main():
    os.makedirs(TEMPLATES_PATH, exist_ok=True)
    
    pipeline = get_test_pipeline()

    for file_name, coordinates_dict in CELL_COORDINATES.items():
        image_path = os.path.join(IMAGES_PATH, file_name)
        sudoku_image = read_image(image_path=image_path)
    
        # BEGIN YOUR CODE
        frontalized_image = frontalize_image(sudoku_image, pipeline)
        sudoku_cells = get_sudoku_cells(frontalized_image)
        
        # END YOUR CODE

        for digit, coordinates in tqdm(coordinates_dict.items(), desc=file_name):
            digit_templates_path = os.path.join(TEMPLATES_PATH, digit)
            os.makedirs(digit_templates_path, exist_ok=True)
            
            digit_template_path = os.path.join(digit_templates_path, f"{os.path.splitext(file_name)[0]}_{digit}.jpg")
            imsave(digit_template_path, sudoku_cells[*coordinates])


if __name__ == "__main__":
    main()
