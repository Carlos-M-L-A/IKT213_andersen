import pathlib, cv2

def print_image_information(image: pathlib.Path):
    img_info = cv2.imread(image, cv2.IMREAD_COLOR)
    if img_info is None:
        print(f"Failed to load image: {image}")
        return

    cv2.namedWindows('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img_info)
    height, width = img_info.shape
    print(f"Image: {image}")
    print(f"Height: {width}, Height: {height}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    script_dir = pathlib.Path(__file__).parent
    image = script_dir / "iris-1.jpg"
    print_image_information(image)

if __name__ == "__main__":
    main()
