import os
import math

import numpy as np
import cv2

from utils import resize_max


def get_image(config, image_path="", camera=None, source="camera"):
    if source == "camera":
        ret, image = camera.read(0)
        if ret:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            image = resize_max(image, config["max_size"])
        else:
            print("ERROR: no image returned from the camera")
            return None
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


def get_existing_image_ids(where_to_save):
    image_ids = {}
    image_files = os.listdir(where_to_save)
    for image_file in image_files:
        image_class, image_id_str = image_file.split("_")
        image_id_str = image_id_str.split(".")[0]
        if image_class not in image_ids:
            image_ids[image_class] = [int(image_id_str)]
        else:
            image_ids[image_class].append(int(image_id_str))

    return image_ids


def create_image_set(
    image,
    hue_range=0.3,
    saturation_range=0.4,
    value_range=0.4,
    yaw_range=20,
    pitch_range=30,
    roll=15,
    scale_range=0.1,
):
    yaw_images = get_random_yaw_images(image, yaw_range, 10)
    pitch_images = get_random_pitch_images(image, yaw_range, 10)

    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hsv_image[..., 0] = np.clip(hsv_image[..., 0] + 11, 0, 255).astype(np.uint8)
    # new_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    images = yaw_images + pitch_images
    return images


def get_random_pitch_images(image, pitch_range, image_num):
    scale = 0.95
    height, width, _ = image.shape
    src_points = np.array(
        [
            [0, 0],
            [width, 0],
            [0, height],
            [width, height],
        ],
        dtype=np.float32,
    )

    pitch_images = []

    pitches = pitch_range * np.random.rand(image_num)
    for i in range(image_num):
        dst_points = []
        dst_points.extend(
            [
                [
                    int((1 - scale) * width),
                    int((1 - math.cos(np.radians(pitches[i]))) * height),
                ],
                [
                    int(scale * width),
                    int((1 - math.cos(np.radians(pitches[i]))) * height),
                ],
                [0, height],
                [width, height],
            ]
        )

        M = cv2.getPerspectiveTransform(
            src_points, np.array(dst_points, dtype=np.float32)
        )

        transformed_image = cv2.warpPerspective(image, M, (width, height))
        pitch_images.append(transformed_image)

        pitch_images.append(transformed_image)

    return pitch_images


def get_random_yaw_images(image, yaw_range, image_num):
    scale = 0.95
    yaw_images = []
    height, width, _ = image.shape
    src_points = np.array(
        [
            [0, 0],
            [width, 0],
            [0, height],
            [width, height],
        ],
        dtype=np.float32,
    )

    yaws = yaw_range * np.random.rand(image_num)
    for i in range(image_num):
        dst_points = []
        dst_points.extend(
            [
                [0, 0],
                [
                    int(math.cos(np.radians(yaws[i])) * width),
                    int((1 - scale) * height),
                ],
                [0, height],
                [int(math.cos(np.radians(yaws[i])) * width), int(scale * height)],
            ]
        )

        M = cv2.getPerspectiveTransform(
            src_points, np.array(dst_points, dtype=np.float32)
        )

        transformed_image = cv2.warpPerspective(image, M, (width, height))
        cv2.imshow("s", image)
        cv2.imshow("r", transformed_image)
        k = cv2.waitKey(0)
        yaw_images.append(transformed_image)

    return yaw_images


def nomalize_image(image, padding=0.1, max_size=500):
    height, width, _ = image.shape
    h_pad = int(padding * width)
    v_pad = int(padding * height)

    padded_image = 255 * np.ones(
        (height + 2 * v_pad, width + 2 * h_pad, 3), dtype=np.uint8
    )

    padded_image[v_pad : v_pad + height, h_pad : h_pad + width, :] = image

    return resize_max(padded_image, max_size)
