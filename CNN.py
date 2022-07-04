from tensorflow import keras
from tensorflow.python.keras import layers

input_size = (640,640,3)

model = keras.Sequential(
    [
        keras.Input(shape=input_size),
        # Conv 1
        layers.Conv2D(96,kernel_size=(11, 11), strides=(4, 4), activation="relu"),
        # Pool 1
        layers.MaxPooling2D(pool_size=(2, 2)),
        # Conv 2
        layers.Conv2D(256,kernel_size=(5, 5), strides=(1,1), activation="relu"),
        layers.ZeroPadding2D(padding=(2, 2)),
        # Pool 2
        layers.MaxPooling2D(pool_size=(2, 2)),
        # Conv 3
        layers.Conv2D(384,kernel_size=(3, 3), strides=(1, 1), activation="relu"),
        layers.ZeroPadding2D(padding=(1, 1)),
        # Conv 4
        layers.Conv2D(384,kernel_size=(3, 3), strides=(1, 1), activation="relu"),
        layers.ZeroPadding2D(padding=(1, 1)),
        # Conv 5
        layers.Conv2D(256,kernel_size=(3, 3), strides=(1, 1), activation="relu"),
        layers.ZeroPadding2D(padding=(1, 1)),
        # Pool 5
        layers.MaxPooling2D(pool_size=(2, 2)),
        # FC 6
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(4096, activation="relu"),
        # FC 7
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(4096, activation="relu"),
        # FC 8
        layers.Dense(1, activation="softmax"),

    ]
)

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])