import numpy as np
import pathlib as path
import cv2
from numpy.core.numeric import ndarray

def print_and_save(title: str, image_to_save: np.ndarray):
    source_path = path.Path(__file__).parent
    cv2.imshow(title, image_to_save)
    cv2.imwrite(source_path / title, image_to_save)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sobel_edge_detection(image: np.ndarray):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (3,3),0) #ksize: (3,3) | SigmaX: 0

    sobel = cv2.Sobel(src=image_blur, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=1)
    print_and_save('Sobel.png', sobel)
    #Problemer med lagring av bildet

def canny_edge_detection(image: np.ndarray, threshold_1: int, threshold_2: int):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (3,3),0) #ksize: (3,3) | SigmaX: 0

    edges = cv2.Canny(image_blur, threshold_1, threshold_2)
    print_and_save('canny.png', edges)

def template_match(image: np.ndarray, template: np.ndarray):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_width, template_height = template_gray.shape

    result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    location = np.where(result >= threshold)
    for part in zip(*location[::-1]):
        cv2.rectangle(image, part, (part[0] + template_width, part[1] + template_height), (0,0,255), 2)

    print_and_save('template_match.png', image)

def resize(image: np.ndarray, scale_factor: int, up_or_down: str):
    rows, cols, channels = image.shape

    result = np.ndarray

    if up_or_down == 'up':
        result = cv2.pyrUp(image, dstsize=(scale_factor * cols, scale_factor * rows))
    elif up_or_down == 'down':
        result = cv2.pyrDown(image, dstsize=(cols//scale_factor, rows//scale_factor))
    else:
        print('Invalid input')

    print_and_save('resize-' + up_or_down + '.png', result)

if __name__ == "__main__":
    source_path = path.Path(__file__).parent
    image_path = source_path / 'lambo.png'
    image = cv2.imread(image_path)

    sobel_edge_detection(image)
    #canny_edge_detection(image, 50, 50)
    #shapes = cv2.imread(source_path / 'shapes-1.png')
    #template = cv2.imread(source_path / 'shapes_template.jpg')
    #template_match(shapes, template)
    #resize(image, 2, 'up')
    #resize(image, 2, 'down')
