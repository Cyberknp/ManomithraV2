import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 50

TRAIN_DIR = r"I:\projects\ManomithraV2\facial\train\train"
TEST_DIR = r"I:\projects\ManomithraV2\facial\test\test"

# ── Data Pipeline (tf.data — no scipy needed) ──────────────────────────────────

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    label_mode="categorical",
)

val_dataset_raw = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="categorical",
)

# ── Align val labels to the train class space ──────────────────────────────────
# Train has 7 classes; val may have fewer. Build a remap so val one-hot vectors
# are padded/reordered to match the full 7-class output of the model.

train_classes = train_dataset.class_names  # e.g. ['angry','disgust',...]
val_classes = val_dataset_raw.class_names  # e.g. ['happy','neutral',...]
NUM_CLASSES = len(train_classes)

print(f"Train classes ({NUM_CLASSES}): {train_classes}")
print(f"Val   classes ({len(val_classes)}): {val_classes}")

# Index in train_classes for each val class (val classes not in train are dropped)
val_to_train_idx = [train_classes.index(c) for c in val_classes if c in train_classes]


def remap_val_labels(images, labels):
    """Convert val one-hot (4-class) → train one-hot (7-class)."""
    # labels shape: (batch, len(val_classes))
    new_labels = tf.zeros((tf.shape(labels)[0], NUM_CLASSES), dtype=tf.float32)
    # Scatter each val column into the correct train column
    for val_i, train_i in enumerate(val_to_train_idx):
        col = labels[:, val_i : val_i + 1]  # (batch, 1)
        padding = [[0, 0], [train_i, NUM_CLASSES - train_i - 1]]
        new_labels = new_labels + tf.pad(col, padding)
    return images, new_labels


val_dataset = val_dataset_raw.map(remap_val_labels, num_parallel_calls=tf.data.AUTOTUNE)

# ── Augmentation + Normalisation (applied only during training) ────────────────

augment = tf.keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(15 / 360),  # 15 degrees expressed as a fraction
        layers.RandomTranslation(0.1, 0.1),
    ],
    name="augmentation",
)

normalise = tf.keras.Sequential(
    [layers.Rescaling(1.0 / 255)],
    name="normalisation",
)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.map(
    lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

val_dataset = val_dataset.map(
    lambda x, y: (normalise(x, training=False), y), num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

# ── Model ──────────────────────────────────────────────────────────────────────


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
            layers.GlobalAveragePooling2D(),
            # Final Dense layers
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


model = build_lightweight_model((IMG_SIZE, IMG_SIZE, 1), NUM_CLASSES)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ── Callbacks ──────────────────────────────────────────────────────────────────

callbacks = [
    # Save the best model based on validation accuracy
    ModelCheckpoint(
        r"I:\projects\ManomithraV2\Models\best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    # Stop early if val_loss does not improve for 10 epochs
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    ),
    # Halve the learning rate when val_loss plateaus for 5 epochs
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    ),
]

# ── Training ───────────────────────────────────────────────────────────────────

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=callbacks,
)

# ── Save final model ───────────────────────────────────────────────────────────

model.save(r"I:\projects\ManomithraV2\Models\emotion_model.keras")
print("\nTraining complete. Model saved to Models/emotion_model.keras")
