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

def create_camera_output_information(save_path: pathlib.Path):
    cam = cv2.VideoCapture(0)
    if cam.isOpened() != True:
        cam.open()

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framerate = cam.get(cv2.CAP_PROP_FPS)

    file_to_write = save_path / 'camera_outputs.txt'

    with open(file_to_write, 'w') as f:
        f.write(f'Framerate: {framerate}\n')
        f.write(f'Frame width: {frame_width}\n')
        f.write(f'Frame height: {frame_height}\n')

    cam.release()

def main():
    script_dir = pathlib.Path(__file__).parent
    image = script_dir / "iris-1.jpg"
    print_image_information(image)
    create_camera_output_information(script_dir)

if __name__ == "__main__":
    main()
