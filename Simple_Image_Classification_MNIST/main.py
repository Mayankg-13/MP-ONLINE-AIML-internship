import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# TensorFlow Keras imports
import tensorflow as tf
from tensorflow.keras import Sequential, layers

def load_and_preprocess_data():
    """Load the MNIST dataset and prepare it for the CNN model."""
    print("--- Task 1: Dataset Loading & Preprocessing ---")
    print("Fetching MNIST dataset...")
    # Load built-in MNIST dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    n_train_samples, h, w = X_train.shape
    n_test_samples, _, _ = X_test.shape
    n_classes = 10
    target_names = [str(i) for i in range(10)]
    
    print(f"Total training samples: {n_train_samples}")
    print(f"Total testing samples: {n_test_samples}")
    print(f"Image dimensions: {h}x{w} (grayscale)")
    print(f"Number of classes: {n_classes}")
    
    # Scale pixel values from [0, 255] to [0.0, 1.0]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    print("Scaled pixel values to range [0, 1]")
        
    # Reshape images to shape (height, width, 1) for Conv2D input
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    print(f"Train features shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test features shape: X_test={X_test.shape}, y_test={y_test.shape}")
    
    return X_train, X_test, y_train, y_test, h, w, target_names, n_classes

def build_cnn_model(input_shape, n_classes):
    """Build a simple Sequential CNN model for digit classification."""
    print("\n--- Task 2: Building CNN Architecture ---")
    model = Sequential([
        layers.Input(shape=input_shape),
        # 1st Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        # 2nd Conv Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        # Flatten and Fully Connected Blocks
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25), # Regularization to prevent overfitting
        layers.Dense(n_classes, activation='softmax') # Output probabilities
    ])
    
    print("CNN Architecture Summary:")
    model.summary()
    return model

def train_and_evaluate(model, X_train, X_test, y_train, y_test, h, w, target_names):
    """Compile, train, and evaluate the CNN model."""
    print("\n--- Task 3 & 4: Training & Evaluation ---")
    
    # Setup paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    print("Training CNN model...")
    epochs = 10
    batch_size = 64
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Predict on test data
    print("\nEvaluating model on test data...")
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    
    # 1. Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 2. Performance Table Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\nOverall Test Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision (weighted): {precision:.4f}")
    print(f"  Recall (weighted):    {recall:.4f}")
    print(f"  F1 Score (weighted):  {f1:.4f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }])
    csv_path = os.path.join(script_dir, "mnist_model_performance.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved metrics to '{csv_path}'")
    
    # Plot 1: Training Curves (Loss & Accuracy)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
    plt.title('CNN Model Accuracy History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='s')
    plt.title('CNN Model Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    training_curves_path = os.path.join(plots_dir, "training_curves.png")
    plt.savefig(training_curves_path)
    plt.close()
    print(f"Saved training curves to '{training_curves_path}'")
    
    # Plot 2: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - MNIST CNN Classification')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to '{cm_path}'")
    
    # Plot 3: Visualise sample test predictions
    plt.figure(figsize=(12, 9))
    # Select 12 random test samples
    indices = np.random.choice(len(X_test), size=12, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(3, 4, i + 1)
        # Reshape back to 2D for plotting
        img_2d = X_test[idx].reshape(h, w)
        plt.imshow(img_2d, cmap='gray')
        
        true_lbl = target_names[y_test[idx]]
        pred_lbl = target_names[y_pred[idx]]
        
        title_color = 'green' if y_test[idx] == y_pred[idx] else 'red'
        plt.title(f"True: {true_lbl}\nPred: {pred_lbl}", 
                  color=title_color, fontsize=11)
        plt.axis('off')
        
    plt.suptitle("Sample Test Predictions (Green = Correct, Red = Incorrect)", fontsize=14)
    plt.tight_layout()
    sample_preds_path = os.path.join(plots_dir, "sample_predictions.png")
    plt.savefig(sample_preds_path)
    plt.close()
    print(f"Saved sample prediction visualization to '{sample_preds_path}'")

def main():
    # Load and preprocess
    X_train, X_test, y_train, y_test, h, w, target_names, n_classes = load_and_preprocess_data()
    
    # Build
    input_shape = (h, w, 1)
    model = build_cnn_model(input_shape, n_classes)
    
    # Train and evaluate
    train_and_evaluate(model, X_train, X_test, y_train, y_test, h, w, target_names)
    
    print("\nMNIST CNN image classification assignment completed successfully!")

if __name__ == "__main__":
    main()
