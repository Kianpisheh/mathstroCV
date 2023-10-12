import cv2
import numpy as np


def get_features(frame, config):
    colors_data, colors_rgb = get_colors(frame)
    color_features = get_color_features(colors_data)
    x = [v for v in color_features.values()]

    return x


def get_colors(frame, output_type="np-array", unified=False):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_thresh = {
        "red": [[0, 100, 20], [15, 255, 255]],
        "green": [[40, 40, 40], [90, 255, 255]],
        "blue": [[91, 50, 50], [130, 255, 255]],
        "yellow": [[16, 100, 100], [40, 255, 255]],
    }

    if unified:
        color_data = np.zeros((frame.shape[0], frame.shape[1]))
    else:
        color_data = {}

    color_data_rgb = {}
    for i, color in enumerate(["red", "green", "blue", "yellow"]):
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

        color_channel_data = cv2.bitwise_and(frame, frame, mask=feat_mask)
        color_value = i + 1 if unified else 1

        color_channel_data_binary = cv2.threshold(
            color_channel_data, 128, color_value, cv2.THRESH_BINARY
        )[1]
        ch = i
        if i == 3:
            ch = 0

        color_data_rgb[color] = color_channel_data.copy()
        if unified:
            color_data = color_data + color_channel_data_binary[:, :, ch]

        else:
            color_data[color] = color_channel_data_binary[:, :, ch]

        if output_type == "list":
            color_data[color] = color_data[color].tolist()

    return color_data, color_data_rgb


def get_color_features(colors_data):
    num_colors = [0, 0, 0, 0]  # RGBY

    for i in range(colors_data.shape[0]):
        for j in range(colors_data.shape[1]):
            if colors_data[i][j] == 1:
                num_colors[0] = num_colors[0] + 1
            elif colors_data[i][j] == 2:
                num_colors[1] = num_colors[1] + 1
            elif colors_data[i][j] == 3:
                num_colors[2] = num_colors[2] + 1
            elif colors_data[i][j] == 4:
                num_colors[3] = num_colors[3] + 1

            # or num_colors[colors_data[i][j]-1] += 1

    # number of all pixels
    N = colors_data.size
    color_feature = [0, 0, 0, 0]

    for i in range(4):
        color_feature[i] = num_colors[i] / N

    return color_feature


def get_edges(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, 100, 200, apertureSize=3)

    grad_x = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)

    grad_dir = np.arctan2(grad_y, grad_x) * 180 / np.pi

    # edge quantization
    edge_dir = {}
    edge_dir["0"] = (grad_dir >= -20 & grad_dir <= 20).astype(np.uint8) * 1
    edge_dir["45"] = (grad_dir >= 35 & grad_dir <= 55).astype(np.uint8) * 2
    edge_dir["90"] = (grad_dir >= 75 & grad_dir <= 105).astype(np.uint8) * 3
    edge_dir["135"] = (grad_dir >= -55 & grad_dir <= -35).astype(np.uint8) * 4

    return edge_dir["0"] + edge_dir["45"] + edge_dir["90"] + edge_dir["135"]


def get_edge_features(edges_data):
    num_edges = [0, 0, 0, 0]  # 0-45-90-135
    # number of all pixels
    N = edges_data.size

    for i in range(edges_data.shape[0]):
        for j in range(edges_data.shape[1]):
            if edges_data[i][j] == 1:
                num_edges[0] = num_edges[0] + 1
            elif edges_data[i][j] == 2:
                num_edges[1] = num_edges[1] + 1
            elif edges_data[i][j] == 3:
                num_edges[2] = num_edges[2] + 1
            elif edges_data[i][j] == 4:
                num_edges[3] = num_edges[3] + 1

    edge_feature = [0, 0, 0, 0]
    for i in range(4):
        edge_feature[i] = num_edges[i] / N

    return edge_feature


# def get_color_features(colors_data):
#     color_features = {}
#     for color in colors_data:
#         nonzeros = [np.count_nonzero(colors_data[color][:, :, i]) for i in range(3)]
#         color_features[color] = (3 * max(nonzeros)) / colors_data[color].size

#     return color_features
