import pathlib as path
import numpy as np
import cv2

def print_and_save(title, image_to_save):
    cv2.imshow(title, image_to_save)
    cv2.imwrite(title, image_to_save)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def padding(image, border_width: int):
    BLACK = [255,255,255]
    image_w_borders = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=BLACK)

    print_and_save('padding.jpg', image_w_borders)

def crop(image, x_0: int, x_1: int, y_0: int, y_1: int):
    rows, cols, channels = image.shape
    cropped_image = image[x_0: rows - x_1, y_0: cols - y_1]

    print_and_save('crop.jpg', cropped_image)

def resize(image, width: int, height: int):
    rows, cols, channels = image.shape
    resized_image = cv2.resize(image, (rows + width, cols + height), interpolation=cv2.INTER_CUBIC)

    print_and_save('resize.jpg', resized_image)

def copy(image, emptyPictureArray):
    #Doen't use cv2.copy() or np.copy()
    emptyPictureArray[:] = image

    print_and_save('copy.jpg', emptyPictureArray)

def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print_and_save('gray.jpg', gray_image)

def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    print_and_save('hsv.jpg', hsv_image)

def hue_shifted(image, emptyPictureArray, hue: np.uint8):
    #When a color value goes beyond 255 or bellow 0 it will overflow to either 0 or 255 respectivelly
    emptyPictureArray[:] = image + 50

    print_and_save('hue_shift.jpg', emptyPictureArray)

def smoothing(image):
    blurred_image = cv2.GaussianBlur(image, (15, 15), sigmaX=0, borderType=cv2.BORDER_DEFAULT)

    print_and_save('smoothing.jpg', blurred_image)

def rotation(image, rotation_angle):
    rotated_image = cv2.rotate(image, rotation_angle)

    print_and_save('rotation.jpg', rotated_image)

if __name__ == '__main__':
    source_path = path.Path(__file__).parent.parent.parent
    image_path = source_path / 'assignment_1' / 'solutions' / 'iris-1.jpg'
    image = cv2.imread(image_path)
    padding(image, 100)
    crop(image, 200, 130, 200, 130)
    resize(image, 200, 200)

    rows, cols, channels = image.shape
    emptyPictureArray = np.zeros((rows, cols, 3), dtype=np.uint8)
    copy(image, emptyPictureArray)
    grayscale(image)
    hsv(image)
    hue_shifted(image, emptyPictureArray, 50)
    smoothing(image)
    rotation(image, cv2.ROTATE_90_CLOCKWISE)
