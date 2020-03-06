import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import cv2
import pickle

SIZE = 150  # Target Size of all the images after resizing
DATA_DIR = "D:/Project/Task/data_sleeves"  # The directory where the script looks for dataset
CLASSES = ["full_sleeves", "half_sleeves", "sleeveless", "three_fourth"]
training_set = []  # list to store traing set


def pipeline():
    # Looping through each class
    for Class in CLASSES:
        path = os.path.join(DATA_DIR, Class)
        class_num = CLASSES.index(Class)

        # Looping through each image
        for img in os.listdir(path):

            # Try catch block to avoid errors cause by broken image or format:
            try:
                arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_arr = cv2.resize(arr, (SIZE, SIZE))
                training_set.append([img_arr, class_num])



            except:
                # print("Invalid image type or directory")
                pass

    # Shuffling the set:
    random.shuffle(training_set)
    print(len(training_set))

    x_feature = []
    y_label = []

    # Creating numpy array for features and sticking labels:
    for feature, label in training_set:
        x_feature.append(feature)
        y_label.append(label)
    x_feature = np.array(x_feature).reshape(-1, SIZE, SIZE, 1)

    # Writing the arrays in binary files:

    file = open("Feature.dat", "wb", )  #
    pickle.dump(x_feature, file)
    file.close()

    file = open("Label.dat", "wb")
    pickle.dump(y_label, file)
    file.close()


pipeline()

# def return_data():
#    return SIZE, CLASSES
