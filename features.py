import cv2
import numpy as np


from utils import resize_max


def get_colors(frame):
    color_data = {}
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_thresh = {
        "red": [[0, 100, 20], [10, 255, 255]],
        "green": [[40, 40, 40], [90, 255, 255]],
        "blue": [[90, 50, 50], [130, 255, 255]],
        "yellow": [[20, 100, 100], [45, 255, 255]],
    }

    # color features
    for color in ["red", "green", "blue", "yellow"]:
        # extract red pixels
        feat_mask = cv2.inRange(
            hsv_image,
            np.array(color_thresh[color][0]),
            np.array(color_thresh[color][1]),
        )

        if color == "red":
            feat_mask = feat_mask | cv2.inRange(
                hsv_image,
                np.array([170, 70, 50]),
                np.array([180, 255, 255]),
            )

        color_data[color] = cv2.bitwise_and(frame, frame, mask=feat_mask)

        # # creating a black border border for the frame
        # black = [128, 128, 128]
        # color_data[color] = cv2.copyMakeBorder(
        #     color_data[color], 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=black
        # )

    return color_data


def get_edges(frame):
    # edge features
    # Apply Canny edge detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    # black = [128, 128, 128]
    # edges = cv2.copyMakeBorder(edges, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=black)
    # diagonal_edge_mask = cv2.copyMakeBorder(
    #     diagonal_edge_mask, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=black
    # )
    edges = cv2.merge((edges, edges, edges))
    diagonal_edge_mask = cv2.merge(
        (diagonal_edge_mask, diagonal_edge_mask, diagonal_edge_mask)
    )

    return {"edges": edges, "45": diagonal_edge_mask}


def get_color_features(colors_data):
    color_features = {}
    for color in colors_data:
        nonzeros = [np.count_nonzero(colors_data[color][:, :, i]) for i in range(3)]
        color_features[color] = (3 * max(nonzeros)) / colors_data[color].size

    return color_features


def get_features(frame):
    # get colors
    colors_data = get_colors(frame)

    color_features = get_color_features(colors_data)
    x = [v for v in color_features.values()]

    return x
