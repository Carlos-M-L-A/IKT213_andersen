import numpy as np
import pathlib as path
import cv2

def print_and_save(title: str, image_to_save: np.ndarray):
    source_path = path.Path(__file__).parent
    cv2.imshow(title, image_to_save)
    cv2.imwrite(source_path / title, image_to_save)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def reference_image(image: np.ndarray):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detection = cv2.cornerHarris(image_gray, 2, 3, 0.04)

    detection = cv2.dilate(detection, None)

    image[detection > 0.01 * detection.max()] = [255,0,0]

    print_and_save('harris.png', image)

def sift(image_to_align: np.ndarray, reference_image: np.ndarray, max_features: int, good_match_percent: np.float16):
    image_to_align_gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)


    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(image_to_align, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(reference_image, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = [m for m in matches if m.distance > good_match_percent]

    result = cv2.drawMatches(image_to_align_gray, keypoints_1, reference_image_gray, keypoints_2, matches[:max_features], reference_image_gray, flags=2)

    print_and_save('sift.jpg', result)

if __name__ == "__main__":
    source_path = path.Path(__file__).parent
    image_path = source_path / 'reference_img.png'
    image = cv2.imread(image_path)

    reference_image(image)

    image2_path = source_path / 'align_this.jpg'
    image2 = cv2.imread(image2_path)
    sift(image2, image, 10, np.float16(0.7))
