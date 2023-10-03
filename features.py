import cv2
import numpy as np


def get_colors(frame):
    features = {}
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

        features[color] = cv2.bitwise_and(frame, frame, mask=feat_mask)

        # creating a black border border for the frame
        black = [128, 128, 128]
        features[color] = cv2.copyMakeBorder(
            features[color], 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=black
        )

    return features
