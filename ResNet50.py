import tensorflow as tf
from tensorflow.python.keras import layers

#tf.keras.datasets.cifar10.load_data()

def preprocess(X, Y):
    X1 = tf.keras.applications.resnet50.preprocess_input(X)
    Y1 = tf.keras.utils.to_categorical(Y,10)
    return X1, Y1

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, y_train = preprocess(x_train, y_train)
x_test, y_test = preprocess(x_test, y_test)

input = tf.keras.Input(shape=(32, 32, 3))
model = tf.keras.applications.resnet50.ResNet50(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
poids = tf.keras.callbacks.ModelCheckpoint(filepath="poids.h5", monitor="val_acc", mode="max", save_best_only=True)
truc = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), callbacks=[poids])
model.summary()
model.save("poids.h5")