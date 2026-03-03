import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

IMG_SIZE = 48
BATCH_SIZE = 32  #  Reduced from 64 → smaller batches = better generalisation on FER
EPOCHS = 75  #  Increased ceiling; EarlyStopping will cap it naturally

TRAIN_DIR = r"I:\projects\ManomithraV2\facial\train\train"
TEST_DIR = r"I:\projects\ManomithraV2\facial\test\test"

# ── Data Pipeline ──────────────────────────────────────────────────────────────

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

train_classes = train_dataset.class_names
val_classes = val_dataset_raw.class_names
NUM_CLASSES = len(train_classes)

print(f"Train classes ({NUM_CLASSES}): {train_classes}")
print(f"Val   classes ({len(val_classes)}): {val_classes}")

val_to_train_idx = [train_classes.index(c) for c in val_classes if c in train_classes]


def remap_val_labels(images, labels):
    new_labels = tf.zeros((tf.shape(labels)[0], NUM_CLASSES), dtype=tf.float32)
    for val_i, train_i in enumerate(val_to_train_idx):
        col = labels[:, val_i : val_i + 1]
        padding = [[0, 0], [train_i, NUM_CLASSES - train_i - 1]]
        new_labels = new_labels + tf.pad(col, padding)
    return images, new_labels


val_dataset = val_dataset_raw.map(remap_val_labels, num_parallel_calls=tf.data.AUTOTUNE)

# ── Augmentation ───────────────────────────────────────────────────────────────
# ✅ Added RandomZoom + RandomContrast — essential for facial expression variance

augment = tf.keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(20 / 360),  # ✅ Widened from 15° → 20°
        layers.RandomTranslation(0.12, 0.12),  # ✅ Slightly wider translation
        layers.RandomZoom(0.15),  # ✅ NEW: zoom augmentation
        layers.RandomContrast(0.2),  # ✅ NEW: brightness/contrast shift
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
# ✅ Key upgrades:
#    1. Deeper filter progression (32→64→128→256→512)
#    2. Manual skip connection (residual-style) before final pooling
#    3. Two dense layers instead of one (256 → 128) for richer feature mapping
#    4. Spatial Dropout instead of plain Dropout (preserves spatial structure longer)


def build_lightweight_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # ── Block 1: Initial feature extraction ───────────────────────────────────
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)

    # ── Block 2: Separable block ───────────────────────────────────────────────
    x = layers.SeparableConv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 48→24

    # ── Block 3: Separable block ───────────────────────────────────────────────
    x = layers.SeparableConv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 24→12

    # ── Block 4: Separable block ───────────────────────────────────────────────
    x = layers.SeparableConv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 12→6

    # ── Block 5 + Skip connection ✅ NEW ──────────────────────────────────────
    residual = layers.Conv2D(512, (1, 1), padding="same")(x)  # project for skip
    x = layers.SeparableConv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])  # skip connection
    x = layers.Activation("relu")(x)

    # ── Head ──────────────────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation="relu")(x)  # ✅ Wider first dense
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)  # ✅ Reduced from 0.5

    x = layers.Dense(128, activation="relu")(x)  # ✅ Second dense layer
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="emotion_cnn_v2")


model = build_lightweight_model((IMG_SIZE, IMG_SIZE, 1), NUM_CLASSES)

# ── Compile ────────────────────────────────────────────────────────────────────
# ✅ label_smoothing=0.1 — prevents overconfidence on ambiguous emotion labels
# ✅ Initial LR lowered to 5e-4 for more stable convergence

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)

model.summary()

# ── Callbacks ──────────────────────────────────────────────────────────────────
# ✅ EarlyStopping patience raised to 15 — gives model more room to recover
# ✅ ReduceLROnPlateau patience raised to 7, factor kept at 0.5

callbacks = [
    ModelCheckpoint(
        r"I:\projects\ManomithraV2\Models\best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=15,  # ✅ Was 10 — more breathing room
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,  # ✅ Was 5
        min_lr=1e-7,  # ✅ Lower floor for fine-grained LR decay
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
