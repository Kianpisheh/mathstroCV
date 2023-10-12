import os
import re

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import cv2

from features import *
from functions import get_image, visualize

image_seq_path = "./images/training"
test_image_seq_path = "./images/"
config = {"max_size": 500, "unified_color": True}
cap = None


# -----------------------Training----------------------#
# Feature extraction
c_X, e_X = [], []
Y = []
images_files = os.listdir(image_seq_path)
for image_file in images_files:
    image = get_image(config, image_seq_path + "/" + image_file, source="image-seq")
    color_data, _ = get_colors(image, unified=True)
    edge_data = get_edges(image)
    c_x = get_color_features(color_data)
    e_x = get_edge_features(edge_data)
    y = parts = re.split(r"\d+", image_file)[0]
    c_X.append(c_x)
    Y.append(y)

# Modeling
model = DecisionTreeClassifier()
model.fit(X, Y)
# ----------------------------------------------------#


# ----------------------Test--------------------------#
i = 0
cap = cv2.VideoCapture("http://10.0.0.200:4747/video")
while True:
    image = get_image(config, camera=cap, source="camera")

    # get color information
    color_data, colors_data_rgb = get_colors(image, unified=True)

    # feature extraction
    x = get_color_features2(color_data)

    # prediction
    pred = model.predict([x])[0]
    print("prediction: ", pred)

    # visualization
    visualize(image, colors_data_rgb)
    k = cv2.waitKey(30)
    if k == 27:
        cv2.destroyAllWindows()
        break

    i += 1
# ---------------------------------------------------------#

# ------------------Get color features---------------------#


# image_filtered = cv2.GaussianBlur(image, (3, 3), 0)
# colors_data, colors_data_rgb = get_colors(image_filtered)

# feature extraction
# color_features = get_color_features(colors_data)
# edge_features = get_edges(image_filtered)


# --------------------Visualization------------------#

# visualize(image, color_images)

# image = cv2.copyMakeBorder(image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[128, 128, 128])
# display images and the extracted features
