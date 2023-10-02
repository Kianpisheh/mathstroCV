import cv2
import numpy as np

cap = cv2.VideoCapture("http://10.0.0.200:4747/video")

color_thresh = {
    "red": [[0, 100, 20], [10, 255, 255]],
    "green": [[40, 40, 40], [90, 255, 255]],
    "blue": [[90, 50, 50], [130, 255, 255]],
    "yellow": [[20, 100, 100], [45, 255, 255]],
}

while True:
    ret, frame = cap.read(0)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.resize(
        frame, ((int(frame.shape[1] * 0.5)), (int(frame.shape[0] * 0.5)))
    )
    frame_filtered = cv2.GaussianBlur(frame, (5, 5), 0)

    hsv_image = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2HSV)

    features = {}
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

        features[color] = cv2.bitwise_and(
            frame_filtered, frame_filtered, mask=feat_mask
        )

        # creating a black border border for the frame
        black = [128, 128, 128]
        features[color] = cv2.copyMakeBorder(
            features[color], 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=black
        )

    # edge features
    # Apply Canny edge detection
    gray_frame = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, 100, 200)
    gradient_x = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
    # Define angle threshold for diagonal edges
    angle_threshold = (33, 60)

    # Create a mask for diagonal edges
    diagonal_edge_mask = (
        (gradient_direction >= angle_threshold[0])
        & (gradient_direction <= angle_threshold[1])
    ).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)

    dilated_image = cv2.dilate(diagonal_edge_mask, kernel, iterations=1)
    diagonal_edge_mask = cv2.bitwise_and(dilated_image, edges)

    frame = cv2.copyMakeBorder(frame, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=black)

    # display frames and the extracted features
    row1 = cv2.hconcat((frame, features["red"], features["green"]))
    row2 = cv2.hconcat((features["blue"], features["yellow"], frame))
    all_iimages = cv2.vconcat((row1, row2))
    cv2.imshow("Original Image", all_iimages)

    cv2.imshow("edges", edges)
    cv2.imshow("edges2", diagonal_edge_mask)

    k = cv2.waitKey(30)
    if k == 27:
        break


# Load an image
# image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

# # Apply the Sobel filter for vertical edge detection
# sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)

# # Convert the resulting image to an absolute value and scale it to the range [0, 255]
# sobel_x = np.absolute(sobel_x)
# sobel_x = np.uint8(255 * sobel_x / np.max(sobel_x))
