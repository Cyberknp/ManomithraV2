from tensorflow.keras import layers, models


def build_lightweight_model(input_shape, num_classes):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            # Initial Conv layer to extract low-level features
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            # Depthwise Separable block 1
            layers.SeparableConv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            # Depthwise Separable block 2
            layers.SeparableConv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            # Depthwise Separable block 3
            layers.SeparableConv2D(256, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),  # This saves massive amounts of memory
            # Final Dense Layers
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


model = build_lightweight_model(
    (48, 48, 1), 7
)  # Adjust classes based on your 'facial' folders
