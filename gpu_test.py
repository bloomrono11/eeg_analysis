import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def main():
    # Simple CNN for example
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Check GPU usage during training

    x_train = np.random.rand(100, 28, 28, 1)
    y_train = np.random.randint(0, 10, 100)

    history = model.fit(x_train, y_train, epochs=5)

    print(f"GPU is being used: {tf.config.experimental.list_physical_devices('GPU')}")


if __name__ == '__main__':
    main()