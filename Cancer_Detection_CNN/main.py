import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# TensorFlow Keras imports
import tensorflow as tf
from tensorflow.keras import Sequential, layers

def generate_synthetic_cell_data(n_samples=1200, img_size=50):
    """Generate synthetic histopathology image patches representing normal vs malignant cells."""
    print(f"Generating {n_samples} synthetic histopathology cell patches...")
    X = np.zeros((n_samples, img_size, img_size, 3), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.int32)
    
    # Background cytoplasm color (Pink/Magenta stain in H&E): RGB around (240, 200, 220)
    bg_color = np.array([0.94, 0.78, 0.86], dtype=np.float32)
    
    # Normal Nucleus color (Light Purple): RGB around (80, 50, 130)
    normal_nucleus_color = np.array([0.31, 0.20, 0.51], dtype=np.float32)
    
    # Cancer Nucleus color (Dark/Hyperchromatic Purple): RGB around (40, 20, 90)
    cancer_nucleus_color = np.array([0.16, 0.08, 0.35], dtype=np.float32)
    
    half_samples = n_samples // 2
    
    for i in range(n_samples):
        # Fill image with pink background cytoplasm
        X[i] = bg_color
        
        # Add random microscopic texture noise
        noise = np.random.normal(0.0, 0.015, (img_size, img_size, 3)).astype(np.float32)
        X[i] = np.clip(X[i] + noise, 0.0, 1.0)
        
        # Determine cell type: first half normal, second half malignant
        if i >= half_samples:
            y[i] = 1 # Cancer
            # Malignant cells: high density, larger, hyperchromatic (darker), irregular overlap nuclei
            num_nuclei = np.random.randint(5, 10)
            for _ in range(num_nuclei):
                cx = np.random.randint(10, img_size - 10)
                cy = np.random.randint(10, img_size - 10)
                # Irregular shape represented by distinct x and y radii
                r_x = np.random.randint(6, 11)
                r_y = np.random.randint(4, 9)
                
                # Draw ellipse nucleus onto numpy array
                y_grid, x_grid = np.ogrid[:img_size, :img_size]
                mask = ((x_grid - cx) / r_x)**2 + ((y_grid - cy) / r_y)**2 <= 1.0
                X[i][mask] = cancer_nucleus_color
        else:
            y[i] = 0 # Normal
            # Normal cells: small, regular, spaced circular nuclei (1-3)
            num_nuclei = np.random.randint(1, 4)
            for _ in range(num_nuclei):
                cx = np.random.randint(15, img_size - 15)
                cy = np.random.randint(15, img_size - 15)
                # Circular shape represented by small, matching x and y radii
                r = np.random.randint(4, 6)
                
                # Draw circular nucleus onto numpy array
                y_grid, x_grid = np.ogrid[:img_size, :img_size]
                mask = ((x_grid - cx) / r)**2 + ((y_grid - cy) / r)**2 <= 1.0
                X[i][mask] = normal_nucleus_color
                
    # Smooth images slightly to simulate microscope focus blur
    for i in range(n_samples):
        # A simple box blur filter implemented in numpy
        for c in range(3):
            # Pad image boundaries to prevent errors
            padded = np.pad(X[i][:, :, c], 1, mode='edge')
            # 3x3 average blur
            X[i][:, :, c] = (
                padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
                padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
            ) / 9.0
            
    print(f"Generated data dimensions: Features={X.shape}, Labels={y.shape}")
    return X, y

def load_and_preprocess_data():
    """Load the synthetic cell dataset and split it stratified into training/testing sets."""
    print("--- Task 1: Dataset Generation & Preprocessing ---")
    X, y = generate_synthetic_cell_data(n_samples=1200, img_size=50)
    
    n_samples, h, w, c = X.shape
    n_classes = 2
    target_names = ["Normal", "Malignant"]
    
    print("\nClass distribution:")
    class_counts = pd.Series(y).value_counts().sort_index()
    for idx, count in class_counts.items():
        print(f"  {target_names[idx]}: {count} patches")
        
    # Split into train and test sets (75% train, 25% test, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\nTrain set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")
    
    return X_train, X_test, y_train, y_test, h, w, c, target_names, n_classes

def build_cnn_model(input_shape):
    """Build a Sequential CNN model for cancer classification."""
    print("\n--- Task 2: Building CNN Architecture ---")
    model = Sequential([
        layers.Input(shape=input_shape),
        # 1st Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        # 2nd Conv Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        # 3rd Conv Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        # Flatten and Fully Connected Blocks
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3), # Regularization to prevent overfitting
        layers.Dense(1, activation='sigmoid') # Sigmoid for binary probability output
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
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    print("Training CNN model...")
    epochs = 15
    batch_size = 32
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
    # Binary predictions: positive if probability > 0.5
    y_pred = (predictions >= 0.5).astype(np.int32).flatten()
    
    # 1. Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 2. Performance Table Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\nOverall Test Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }])
    csv_path = os.path.join(script_dir, "cancer_model_performance.csv")
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
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Cancer Cell Detection')
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
        plt.imshow(X_test[idx])
        
        true_lbl = target_names[y_test[idx]]
        pred_lbl = target_names[y_pred[idx]]
        pred_prob = predictions[idx][0]
        
        title_color = 'green' if y_test[idx] == y_pred[idx] else 'red'
        plt.title(f"True: {true_lbl}\nPred: {pred_lbl} ({pred_prob:.2f})", 
                  color=title_color, fontsize=10)
        plt.axis('off')
        
    plt.suptitle("Sample Test Predictions (Green = Correct, Red = Incorrect)", fontsize=14)
    plt.tight_layout()
    sample_preds_path = os.path.join(plots_dir, "sample_predictions.png")
    plt.savefig(sample_preds_path)
    plt.close()
    print(f"Saved sample prediction visualization to '{sample_preds_path}'")

def main():
    # Load and preprocess
    X_train, X_test, y_train, y_test, h, w, c, target_names, n_classes = load_and_preprocess_data()
    
    # Build
    input_shape = (h, w, c)
    model = build_cnn_model(input_shape)
    
    # Train and evaluate
    train_and_evaluate(model, X_train, X_test, y_train, y_test, h, w, target_names)
    
    print("\nCancer Detection CNN assignment completed successfully!")

if __name__ == "__main__":
    main()
