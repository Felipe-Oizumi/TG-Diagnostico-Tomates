import tensorflow as tf


IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
CHANNELS = 3

NUM_CLASSES = 10


def create_cnn():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
                CHANNELS
            )
        ),

        tf.keras.layers.Rescaling(
            1.0 / 255
        ),

        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])


    model.compile(
        optimizer="adam",
        loss=(
            "sparse_"
            "categorical_crossentropy"
        ),
        metrics=[
            "accuracy"
        ]
    )

    return model