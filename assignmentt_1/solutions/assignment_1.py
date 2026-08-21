import pathlib
import cv2


def print_image_information(image: pathlib.Path):
    img_info = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    if img_info is None:
        print(f"Failed to load image in location: {image}")
        return

    height, width, channels = img_info.shape
    print(f"Height: {width}, Height: {height}")
    print(f'Channels = {channels}')
    print(f'Size: {img_info.size}')
    print(f'Data type {img_info.dtype}')

    cv2.imshow('image', img_info)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    script_dir = pathlib.Path(__file__).parent
    image = script_dir / "iris-1.jpg"
    print_image_information(image)

if __name__ == "__main__":
    main()
