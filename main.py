import os
import re

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import cv2

from utils import resize_max
from features import *

cap = cv2.VideoCapture("http://10.0.0.200:4747/video")


image_seq_path = "./images/training"
test_image_seq_path = "./images/test"
source = "image-seq"

# training
X = []
Y = []
images_files = os.listdir(image_seq_path)
for image_file in images_files:
    frame = cv2.imread(image_seq_path + "/" + image_file)
    frame = resize_max(frame, 500)
    x = get_features(frame)
    y = parts = re.split(r"\d+", image_file)[0]

    X.append(x)
    Y.append(y)

# training
model = DecisionTreeClassifier()
model.fit(X, Y)

model2 = LogisticRegression()
model2.fit(X, Y)

i = 0
features = []
images_files = os.listdir(test_image_seq_path)
while True:
    if i == len(images_files):
        break
    if source == "phone-camera":
        ret, frame = cap.read(0)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.resize(
            frame, ((int(frame.shape[1] * 0.7)), (int(frame.shape[0] * 0.7)))
        )
    elif source == "image-seq":
        frame = cv2.imread(test_image_seq_path + "/" + images_files[i])
        frame = resize_max(frame, 500)
        i += 1

    x = get_features(frame)
    print("decision tree: ", model.predict([x])[0])
    print("logistic regression: ", model2.predict([x])[0])

    frame_filtered = cv2.GaussianBlur(frame, (11, 11), 0)
    colors_data = get_colors(frame_filtered)

    # feature extraction
    color_features = get_color_features(colors_data)
    edge_features = get_edges(frame_filtered)

    # frame = cv2.copyMakeBorder(frame, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[128, 128, 128])

    # display frames and the extracted features
    row1 = cv2.hconcat(
        (frame, colors_data["red"], colors_data["green"], colors_data["blue"])
    )
    row2 = cv2.hconcat(
        (colors_data["yellow"], edge_features["edges"], edge_features["45"], frame)
    )
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
