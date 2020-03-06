import tensorflow as tf
import cv2

SIZE = 150
CLASSES = ["full_sleeves", "half_sleeves", "sleeveless", "three_fourth"]


def load_test_img(path):  # Fucntion to load the image for testing, it takes image path as argument
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(arr, (SIZE, SIZE))

    return img.reshape(-1, SIZE, SIZE, 1)


model = tf.keras.models.load_model("CNN-64x3.model")        # Loading the previously saved model.

Output = model.predict([load_test_img("im__29.png")])   # Classifying the image by passing the name of test image

print(CLASSES[int(Output[0][0])])   # Printing the output on terminal
