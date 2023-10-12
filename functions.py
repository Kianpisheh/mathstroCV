import cv2

from utils import resize_max


def get_image(config, image_path="", camera=None, source="camera"):

    if source == "camera":
        ret, image = camera.read(0)
        if ret:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            image = resize_max(image, config["max_size"])
        else:
            raise Exception("ERROR: no image returned from the camera")
    elif source == "image-seq":
        image = cv2.imread(image_path)
    
    if image.size:
        image = resize_max(image, config["max_size"])


    return image

def visualize(image, colors_data_rgb):

    row1 = cv2.hconcat((image, colors_data_rgb["red"], colors_data_rgb["green"]))
    row2 = cv2.hconcat(
        (colors_data_rgb["blue"], colors_data_rgb["yellow"], colors_data_rgb["yellow"])
    )
    all_iimages = cv2.vconcat((row1, row2))
    cv2.imshow("Original Image", all_iimages)
    

