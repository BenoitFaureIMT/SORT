import tensorflow as tf
import cv2
import numpy as np
from numpy.linalg import norm

model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="weights_resnet.h5",
    input_shape=(224,224,3),
    pooling="avg"
)
model.summary()

def get_pred(im_name):
    image = cv2.imread(im_name)
    image = cv2.resize(image, (224, 224), cv2.INTER_LINEAR)/255
    image = np.array([image])
    return model.predict(image)

def get_dist(im1, im2):
    d = im1[0] - im2[0]
    return norm(d)

def _cosine_distance(a, b, data_is_normalized=False):
    a = a[0][-1]
    b = b[0][-1]
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)

# print(_cosine_distance(get_pred("pic1_1.png"), get_pred("Elephant10.jpg")))

print("-----Picture 1-----")
print(get_dist(get_pred("pic1_1.png"), get_pred("pic1_2.png")))
print("-----Picture 2-----")
print(get_dist(get_pred("pic2_1.png"), get_pred("pic2_2.png")))
print("-----Picture 1/2-----")
print(get_dist(get_pred("pic2_1.png"), get_pred("pic1_2.png")))
print("-----Picture 1/W-----")
print(get_dist(get_pred("pic2_1.png"), get_pred("woof.png")))