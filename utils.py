import cv2


def resize_max(frame, max_size):
    max_dim = max(frame.shape[0], frame.shape[1])

    scale_h = frame.shape[0] / max_dim
    scale_w = frame.shape[1] / max_dim

    if scale_h == 1:
        scale = max_size / frame.shape[0]
    elif scale_w == 1:
        scale = max_size / frame.shape[1]

    return cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
