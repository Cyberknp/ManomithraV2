import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Standardizing image size for a lightweight footprint
IMG_SIZE = 48

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(
    "I:\projects\ManomithraV2\facial\train\train",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
)
