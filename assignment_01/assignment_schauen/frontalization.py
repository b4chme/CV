import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

from utils import read_image, show_image

# BEGIN YOUR IMPORTS

# END YOUR IMPORTS


def find_edges(image):
    """
    Args:
        image (np.array): (grayscale) image of shape [H, W]
    Returns:
        edges (np.array): binary mask of shape [H, W]
    """
    # BEGIN YOUR CODE

    edges = cv2.Canny(image, 50, 150) # 100, 200
    
    # END YOUR CODE

    return edges


def highlight_edges(edges):
    """
    Args:
        edges (np.array): binary mask of shape [H, W]
    Returns:
        highlited_edges (np.array): binary mask of shape [H, W]
    """
    # BEGIN YOUR CODE

    kernel = np.ones((3,3), np.uint8)
    highlited_edges = cv2.dilate(edges, kernel, iterations=1)
    
    
    # END YOUR CODE

    return highlited_edges


def find_contours(edges):
    """
    Args:
        edges (np.array): binary mask of shape [H, W]
    Returns:
        contours (np.array, np.array, ...): tuple of arrays of contours, where each contour is an array of points of shape [N, 1, 2]
    """
    # BEGIN YOUR CODE

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # END YOUR CODE

    return contours


def get_max_contour(contours):
    """
    Args:
        contours (np.array, np.array, ...): tuple of arrays of contours, where each contour is an array of points of shape [N, 1, 2]
    Returns:
        max_contour (np.array): an array of points (vertices) of the contour with the maximum area of shape [N, 1, 2]
    """
    # BEGIN YOUR CODE

    max_contour = max(contours, key=cv2.contourArea)
    
    # END YOUR CODE

    return max_contour


def order_corners(corners):
    """
    Args:
        corners (np.array): an array of corner points (corners) of shape [4, 2]
    Returns:
        ordered_corners (np.array): an array of corner points in order [top left, top right, bottom right, bottom left]
    """
    # BEGIN YOUR CODE

    corners_sum = corners.sum(axis=1)
    top_left = corners[np.argmin(corners_sum)]
    bottom_right = corners[np.argmax(corners_sum)]

    corners_diff = np.diff(corners, axis=1)
    top_right = corners[np.argmin(corners_diff)]
    bottom_left = corners[np.argmax(corners_diff)]
    
    # END YOUR CODE

    ordered_corners = np.array([top_left, top_right, bottom_right, bottom_left])

    return ordered_corners


def find_corners(contour, accuracy=0.1):
    """
    Args:
        contour (np.array): an array of points (vertices) of the contour of shape [N, 1, 2]
        accuracy (float): how accurate the contour approximation should be
    Returns:
        ordered_corners (np.array): an array of corner points (corners) of quadrilateral approximation of contour of shape [4, 2]
                                    in order [top left, top right, bottom right, bottom left]
    """
    # BEGIN YOUR CODE

    epsilon = accuracy * cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)

    
    # END YOUR CODE

    # to avoid errors
    if len(corners) != 4:
        corners = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]])

    ordered_corners = order_corners(corners)

    return ordered_corners


def rescale_image(image, scale=0.5):
    """
    Args:
        image (np.array): input image
        scale (float): scale factor
    Returns:
        rescaled_image (np.array): 8-bit (with range [0, 255]) rescaled image
    """
    # BEGIN YOUR CODE

    height, width = image.shape[:2]
    rescaled_image = cv2.resize(image, (int(width * scale), int(height * scale)))
    
    # END YOUR CODE
    
    return rescaled_image


def gaussian_blur(image, sigma):
    """
    Args:
        image (np.array): input image
        sigma (float): standard deviation for Gaussian kernel
    Returns:
        blurred_image (np.array): 8-bit (with range [0, 255]) blurred image
    """
    # BEGIN YOUR CODE

    ksize = int(6*sigma) + 1 if int(6*sigma) % 2 == 1 else int(6*sigma + 1 )
    blurred_image = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    
    # END YOUR CODE
    
    return blurred_image


def distance(point1, point2):
    """
    Args:
        point1 (np.array): n-dimensional vector
        point2 (np.array): n-dimensional vector
    Returns:
        distance (float): Euclidean distance between point1 and point2
    """
    # BEGIN YOUR CODE

    distance = np.linalg.norm(point1 - point2)
    
    # END YOUR CODE
    
    return distance


def warp_image(image, ordered_corners): 
    """
    Args:
        image (np.array): input image
        ordered_corners (np.array): corners in order [top left, top right, bottom right, bottom left]
    Returns:
        warped_image (np.array): warped with a perspective transform image of shape [H, H]
    """
    # 4 source points
    top_left, top_right, bottom_right, bottom_left = ordered_corners

    # BEGIN YOUR CODE

    # the side length of the Sudoku grid based on distances between corners
    side = max([
        distance(top_left, top_right),
        distance(top_right, bottom_right),
        distance(bottom_right, bottom_left),
        distance(bottom_left, top_left)
    ])

    # what are the 4 target (destination) points?
    destination_points = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]
    ], dtype='float32')

    # perspective transformation matrix
    transform_matrix = cv2.getPerspectiveTransform(ordered_corners.astype('float32'), destination_points)

    # image warped using the found perspective transformation matrix
    warped_image = cv2.warpPerspective(image, transform_matrix, (int(side), int(side)))
    
    # END YOUR CODE

    assert warped_image.shape[0] == warped_image.shape[1], "height and width of the warped image must be equal"

    return warped_image


def frontalize_image(sudoku_image, pipeline):
    """
    Args:
        sudoku_image (np.array): input Sudoku image
        pipeline (Pipeline): Pipeline instance
    Returns:
        frontalized_image (np.array): warped with a perspective transform image of shape [H, H]
    """
    # BEGIN YOUR CODE
    image, ordered_corners = pipeline(sudoku_image)
    frontalized_image = warp_image(image, ordered_corners)
    
    # END YOUR CODE

    return frontalized_image


def show_frontalized_images(image_paths, pipeline, figsize=(16, 12)):
    nrows = len(image_paths) // 4 + 1
    ncols = 4
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if len(axes.shape) == 1:
        axes = axes[np.newaxis, ...]

    for j in range(len(image_paths), nrows * ncols):
        axis = axes[j // ncols][j % ncols]
        show_image(np.ones((1, 1, 3)), axis=axis)
    
    for i, image_path in enumerate(tqdm(image_paths)):
        axis = axes[i // ncols][i % ncols]
        axis.set_title(os.path.split(image_path)[1])
        
        sudoku_image = read_image(image_path=image_path)
        frontalized_image = frontalize_image(sudoku_image, pipeline)

        show_image(frontalized_image, axis=axis, as_gray=True)
