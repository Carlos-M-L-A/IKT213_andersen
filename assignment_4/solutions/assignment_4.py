import pathlib as path

import cv2
import numpy as np


def print_and_save(title: str, image_to_save: np.ndarray):
    source_path = path.Path(__file__).parent
    cv2.imshow(title, image_to_save)
    cv2.imwrite(str(source_path / title), image_to_save)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def reference_image(image: np.ndarray):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detection = cv2.cornerHarris(image_gray, 2, 3, 0.04)

    detection = cv2.dilate(detection, None)

    image[detection > 0.01 * detection.max()] = [255,0,0]

    print_and_save('harris.png', image)

def sift_and_stitch(image_to_align: np.ndarray, reference_image: np.ndarray, max_features: int = 10, good_match_percent: float = 0.7):
    image_to_align_gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # Detect SIFT features
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image_to_align_gray, None)
    kp2, des2 = sift.detectAndCompute(reference_image_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = [m for m in matches if m.distance > good_match_percent]
    matches_to_draw = matches[:max_features]

    result = cv2.drawMatches(
        image_to_align_gray, kp1,
        reference_image_gray, kp2,
        matches_to_draw,
        None,
        flags=2,
    )
    print_and_save('sift.jpg', result)

    # Extract location of good matches
    src_pts = np.array([kp1[m.queryIdx].pt for m in matches_to_draw], dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.array([kp2[m.trainIdx].pt for m in matches_to_draw], dtype=np.float32).reshape(-1, 1, 2)

    # Compute homography
    H, _mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp image_to_align to the reference frame
    h, w = reference_image.shape[:2]
    warped = cv2.warpPerspective(image_to_align, H, (w, h))

    # Simple linear blending (average where both images overlap)
    blended = cv2.addWeighted(reference_image, 0.5, warped, 0.5, 0)

    print_and_save('stitched.jpg', blended)



if __name__ == "__main__":
    source_path = path.Path(__file__).parent
    image_path = source_path / 'reference_img.png'
    image = cv2.imread(str(image_path))

    #reference_image(image)

    image2_path = source_path / 'align_this.jpg'
    image2 = cv2.imread(str(image2_path))

    sift_and_stitch(image2, image, max_features=10, good_match_percent=0.7)
