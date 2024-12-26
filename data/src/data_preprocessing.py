# src/data_preprocessing.py

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_and_preprocess_data(segmented_dir, img_size=(224, 224), batch_size=16, validation_split=0.2):
    """
    Load images from the segmented directory, apply preprocessing and data augmentation.

    Args:
        segmented_dir (str): Path to the segmented images directory.
        img_size (tuple): Target size for images.
        batch_size (int): Batch size for generators.
        validation_split (float): Fraction of data to be used for validation.

    Returns:
        train_generator, validation_generator: Keras ImageDataGenerators for training and validation.
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )

    train_generator = datagen.flow_from_directory(
        segmented_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = datagen.flow_from_directory(
        segmented_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, validation_generator

def visualize_sample_images(generator, class_indices, num_samples=5):
    """
    Visualize a sample of augmented images from the generator.

    Args:
        generator (ImageDataGenerator): Keras ImageDataGenerator.
        class_indices (dict): Mapping of class names to indices.
        num_samples (int): Number of samples to visualize.
    """
    class_labels = list(class_indices.keys())
    plt.figure(figsize=(15, 15))
    for i in range(num_samples):
        img, label = next(generator)
        label_name = class_labels[np.argmax(label)]
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img[0])
        plt.title(label_name)
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Define absolute path to the segmented directory
    SEGMENTED_DIR = r"D:\my Data\OneDrive\Documents\Uni work 5\ANN Lab\ANN Project\data\segmented"

    # Check if the directory exists
    if not os.path.isdir(SEGMENTED_DIR):
        raise FileNotFoundError(f"The directory {SEGMENTED_DIR} does not exist. Please check the path.")

    train_gen, val_gen = load_and_preprocess_data(SEGMENTED_DIR)

    # Optional: Visualize some sample images
    # visualize_sample_images(train_gen, train_gen.class_indices)