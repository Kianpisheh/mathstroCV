import os

import cv2
import numpy as np

from utils import resize_max
from features import get_colors

cap = cv2.VideoCapture("http://10.0.0.200:4747/video")


image_seq_path = "./images"
source = "image-seq"
images_files = os.listdir(image_seq_path)

i = 0
while True:
    if i == len(images_files):
        break
    if source == "video":
        ret, frame = cap.read(0)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.resize(
            frame, ((int(frame.shape[1] * 0.7)), (int(frame.shape[0] * 0.7)))
        )
    elif source == "image-seq":
        frame = cv2.imread(image_seq_path + "/" + images_files[i])
        frame = resize_max(frame, 500)
        i += 1

    frame_filtered = cv2.GaussianBlur(frame, (11, 11), 0)
    features = get_colors(frame_filtered)

    # edge features
    # Apply Canny edge detection
    gray_frame = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, 100, 200, apertureSize=3)
    gradient_x = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
    # Define angle threshold for diagonal edges
    angle_threshold = (35, 55)

    # Apply the Hough Line Transform
    lines = cv2.HoughLines(
        edges, 1.5, np.pi / 180, threshold=80
    )  # You can adjust the threshold

    # Draw the detected lines on the original image
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw the line in red

    # Create a mask for diagonal edges
    diagonal_edge_mask = (
        (gradient_direction >= angle_threshold[0])
        & (gradient_direction <= angle_threshold[1])
    ).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)

    dilated_image = cv2.dilate(diagonal_edge_mask, kernel, iterations=1)
    diagonal_edge_mask = cv2.bitwise_and(dilated_image, edges)
    black = [128, 128, 128]
    edges = cv2.copyMakeBorder(edges, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=black)
    diagonal_edge_mask = cv2.copyMakeBorder(
        diagonal_edge_mask, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=black
    )
    edges = cv2.merge((edges, edges, edges))
    diagonal_edge_mask = cv2.merge(
        (diagonal_edge_mask, diagonal_edge_mask, diagonal_edge_mask)
    )

    frame = cv2.copyMakeBorder(frame, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=black)

    # display frames and the extracted features
    row1 = cv2.hconcat((frame, features["red"], features["green"], features["blue"]))
    row2 = cv2.hconcat((features["yellow"], edges, diagonal_edge_mask, frame))
    all_iimages = cv2.vconcat((row1, row2))
    cv2.imshow("Original Image", all_iimages)

    if source == "image-seq":
        k = cv2.waitKey(0)
        if k != 27:
            continue
        else:
            cv2.destroyAllWindows()
            break

    elif source == "video":
        k = cv2.waitKey(30)
        if k == 27:
            cv2.destroyAllWindows()
            break
