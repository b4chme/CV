import numpy as np
import matplotlib.pyplot as plt

from utils import show_image, show_contours, show_corners
from utils import read_image
import cv2
from frontalization import rescale_image, gaussian_blur

from frontalization import find_edges, highlight_edges
from frontalization import find_contours, get_max_contour
from frontalization import find_corners


def get_test_pipeline():
    pipeline = Pipeline([
        rescale_image,
        gaussian_blur,
        find_edges,
        highlight_edges,
        find_contours,
        get_max_contour,
        find_corners
    ], {
        "rescale_image": {"scale_factor": 0.5},
        "gaussian_blur": {"kernel_size": (5, 5), "sigma": 0.5},
        "find_edges": {"threshold1": 50, "threshold2": 150},
        "highlight_edges": {}, 
        "find_contours": {"mode": cv2.RETR_EXTERNAL, "method": cv2.CHAIN_APPROX_SIMPLE},
        "get_max_contour": {},
        "find_corners": {"max_corners": 4, "quality_level": 0.01, "min_distance": 10}
    })
    return pipeline


class Pipeline(object):
    def __init__(self, functions, parameters={}):
        self.functions = functions
        self.parameters = parameters

    def __call__(self, image, plot=False, figsize=(18, 12)):
        output = image.copy()
        if plot:
            nrows = len(self.functions) // 3 + 1
            ncols = 3
            figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            if len(axes.shape) == 1:
                axes = axes[np.newaxis, ...]
            
            show_image(output, axis=axes[0][0], as_gray=True)
            axes[0][0].set_title("Sudoku image", fontsize=16)

            for j in range(len(self.functions) + 1, nrows * ncols):
                axis = axes[j // ncols][j % ncols]
                show_image(np.ones((1, 1, 3)), axis=axis)
        
        for i, function in enumerate(self.functions):
            kwargs = self.parameters.get(function.__name__, {})
            output = function(output, **kwargs)
            
            if "rescale" in function.__name__:
                image = output.copy()
            
            if plot:
                axis = axes[(i + 1) // ncols][(i + 1) % ncols]
                if "contour" in function.__name__:
                    show_contours(image, contours=output, axis=axis)
                elif "corner" in function.__name__:
                    show_corners(image, corners=output, axis=axis)
                else:
                    if "edge" in function.__name__:
                        show_image(np.bitwise_not(output), axis=axis, as_gray=True)
                    else:
                        show_image(output, axis=axis, as_gray=True)
                
                title = " ".join(function.__name__.split("_"))
                if kwargs:
                    title += " | " + ",".join([f"{key}: {value}" for key, value in kwargs.items()])
                axis.set_title(title, fontsize=16)

        return image, output
