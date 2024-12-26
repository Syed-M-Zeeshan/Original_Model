# src/model_training.py

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import logging
from tensorflow.keras.metrics import Precision, Recall

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def build_model(input_shape, num_classes):
    """
    Build the Convolutional Neural Network model.

    Args:
        input_shape (tuple): Shape of input images.
        num_classes (int): Number of output classes.

    Returns:
        model: Compiled Keras model.
    """
    model = models.Sequential([
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
    )
    
    return model

def plot_training_history(history):
    """
    Plot training and validation accuracy, loss, precision, and recall.

    Args:
        history: Keras History object.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    precision = history.history['precision']
    recall = history.history['recall']
    val_precision = history.history['val_precision']
    val_recall = history.history['val_recall']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, precision, 'b', label='Training Precision')
    plt.plot(epochs, val_precision, 'r', label='Validation Precision')
    plt.title('Training and Validation Precision')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, recall, 'b', label='Training Recall')
    plt.plot(epochs, val_recall, 'r', label='Validation Recall')
    plt.title('Training and Validation Recall')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, val_generator):
    """
    Evaluate the model on the validation set and print classification metrics.

    Args:
        model: Trained Keras model.
        val_generator: Validation data generator.
    """
    val_steps = val_generator.samples // val_generator.batch_size
    Y_pred = model.predict(val_generator, steps=val_steps+1)
    y_pred = np.argmax(Y_pred, axis=1)
    
    logging.info('Confusion Matrix')
    cm = confusion_matrix(val_generator.classes, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=val_generator.class_indices.keys(),
                yticklabels=val_generator.class_indices.keys())
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
    
    logging.info('Classification Report')
    target_names = list(val_generator.class_indices.keys())
    print(classification_report(val_generator.classes, y_pred, target_names=target_names))

if __name__ == "__main__":
    # Suppress TensorFlow INFO and WARNING logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow as tf
    from data_preprocessing import load_and_preprocess_data

    # Define absolute paths
    SEGMENTED_DIR = r"D:\my Data\OneDrive\Documents\Uni work 5\ANN Lab\ANN Project\data\segmented"
    MODEL_DIR = r"D:\my Data\OneDrive\Documents\Uni work 5\ANN Lab\ANN Project\models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Check if the segmented directory exists
    if not os.path.isdir(SEGMENTED_DIR):
        logging.error(f"The directory {SEGMENTED_DIR} does not exist. Please check the path.")
        raise FileNotFoundError(f"The directory {SEGMENTED_DIR} does not exist. Please check the path.")

    # Load and preprocess data
    logging.info("Loading and preprocessing data...")
    train_gen, val_gen = load_and_preprocess_data(SEGMENTED_DIR)
    logging.info("Data loaded successfully.")

    input_shape = train_gen.image_shape  # e.g., (224, 224, 3)
    num_classes = len(train_gen.class_indices)

    # Build model
    logging.info("Building the model...")
    model = build_model(input_shape, num_classes)
    model.summary()

    # Define Callbacks
    checkpoint_path = os.path.join(MODEL_DIR, 'best_model.keras')  # Using .keras extension
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    # Train model
    logging.info("Starting model training...")
    history = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=callbacks
    )
    logging.info("Model training completed.")

    # Plot training history
    plot_training_history(history)

    # Evaluate model
    evaluate_model(model, val_gen)
    logging.info("Model evaluation completed.")