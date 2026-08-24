import pathlib as path
import numpy as np
from matplotlib import pyplot as plt
import cv2


def padding(image_path: path.Path, border_width: int):
    BLACK = [255,255,255]
    image = cv2.imread(image_path)
    image_w_borders = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=BLACK)

    cv2.imshow('image with borders', image_w_borders)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    source_path = path.Path(__file__).parent.parent.parent
    image_path = source_path / 'assignment_1' / 'solutions' / 'iris-1.jpg'
    padding(image_path, 100)
