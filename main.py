import os
import re

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

import cv2

from features import *
from functions import *

training_images = "./images/training"
test_image_seq_path = "./images/test"
config = {"max_size": 500, "unified_color": True}
cap = None


# -------------------Creating image sets---------------------#
source_images = "./images/source_images"
image_files = os.listdir(source_images)
for image_file in image_files:
    image = cv2.imread(f"{source_images}/{image_file}")
    # add border and resize
    image = nomalize_image(image, padding=0.2, max_size=500)
    cv2.imshow("2", image)
    cv2.waitKey(0)
    transformed_images = create_image_set(image)
    cv2.destroyAllWindows()


# -----------------------------------------------------------#

# --------------------Data collection-------------------#
where_to_save = "./images/training2"
# If it doesn't exist, create it
if not os.path.exists(where_to_save):
    os.mkdir(where_to_save)

# find the maximum captured image id for each class
images_ids = get_existing_image_ids(where_to_save)

data_collection_completed = False
cap = cv2.VideoCapture("http://10.0.0.200:4747/video")
while not data_collection_completed:
    image = get_image(config, camera=cap, source="camera")
    if image is None:
        continue

    cv2.imshow("Data collection", image)
    k = cv2.waitKey(30)
    if k == 27:
        cv2.destroyAllWindows()
        data_collection_completed = True
        break
    elif k == 32:
        image_class = "stop"
        new_id = 1
        if image_class in images_ids:
            new_id = max(images_ids[image_class]) + 1
            images_ids[image_class].append(new_id)
        else:
            images_ids[image_class] = [1]
        cv2.imwrite(f"{where_to_save}/{image_class}_{new_id}.jpg", image)

# ----------------------------------------------------#


# -----------------------Training----------------------#
# Feature extraction
c_X, e_X, ce_X = [], [], []
Y = []
images_files = os.listdir(image_seq_path)
for image_file in images_files:
    image = get_image(config, image_seq_path + "/" + image_file, source="image-seq")
    color_data, _ = get_colors(image, unified=True)
    edge_data = get_edges(image)
    c_x = get_color_features(color_data)
    c_X.append(c_x)
    e_x = get_edge_features(edge_data)
    e_X.append(c_x)
    ce_X.append(c_x + e_x)
    y = re.split(r"\d+", image_file)[0]
    Y.append(y)

# Modeling
model = DecisionTreeClassifier()
model.fit(c_X, Y)
model2 = DecisionTreeClassifier()
model2.fit(e_X, Y)
model3 = DecisionTreeClassifier()
model3.fit(ce_X, Y)
# ----------------------------------------------------#

# --------------------Test (image)---------------------#
images_files = os.listdir(test_image_seq_path)
for image_file in images_files:
    image = get_image(
        config, test_image_seq_path + "/" + image_file, source="image-seq"
    )
    color_data, _ = get_colors(image, unified=True)
    edge_data = get_edges(image)
    c_x = get_color_features(color_data)
    pred = model.predict([c_x])
    y = re.split(r"\d+", image_file)[0]
    print(f"prediction: {pred}, actual: {y}")

    e_x = get_edge_features(edge_data)
    pred = model2.predict([e_x])
    print(f"prediction2: {pred}, actual: {y}")

    pred = model3.predict([c_x + e_x])
    print(f"prediction3: {pred}, actual: {y}\n")

# ----------------------Test--------------------------#
# i = 0
# cap = cv2.VideoCapture("http://10.0.0.200:4747/video")
# while True:
#     image = get_image(config, camera=cap, source="camera")

#     # get color information
#     color_data, colors_data_rgb = get_colors(image, unified=True)

#     # feature extraction
#     x = get_color_features2(color_data)

#     # prediction
#     pred = model.predict([x])[0]
#     print("prediction: ", pred)

#     # visualization
#     visualize(image, colors_data_rgb)
#     k = cv2.waitKey(30)
#     if k == 27:
#         cv2.destroyAllWindows()
#         break

#     i += 1
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
