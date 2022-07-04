import tensorflow as tf
from tensorflow.python.keras import layers
import cv2
import numpy as np

model = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(64,64,3),
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)
model.summary()

image = cv2.imread("Elephant10.jpg")
image = cv2.resize(image, (64, 64), cv2.INTER_LINEAR)/255
image = np.array([image])
image = tf.keras.applications.resnet50.preprocess_input(image)
result = model.predict(image)
#print(result[-1,-1])
print(result.shape)