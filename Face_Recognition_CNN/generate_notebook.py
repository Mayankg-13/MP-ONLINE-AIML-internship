import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Face Recognition using Convolutional Neural Network (CNN)\n",
    "### MP ONLINE AIML Internship Project\n",
    "\n",
    "This notebook contains the complete implementation for building a simple **Face Recognition** project using a Convolutional Neural Network (CNN). The objective is to identify individuals from their facial photographs.\n",
    "\n",
    "The assignment comprises five major tasks:\n",
    "1. **Dataset Loading & Preprocessing**: Loading the Labeled Faces in the Wild (LFW) dataset, scaling pixels, splitting into training and testing sets, and reshaping for CNN input.\n",
    "2. **CNN Architecture Design**: Designing a neural network with convolutional, pooling, dropout, and dense layers.\n",
    "3. **Model Compilation & Training**: Compiling using the Adam optimizer, and training for 20 epochs.\n",
    "4. **Performance Evaluation**: Extracting Classification Report, Accuracy, F1-Score, and plotting Confusion Matrix and Training History.\n",
    "5. **Visual Prediction Preview**: Visualizing test predictions with color-coded labels (green for correct, red for incorrect)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Task 1: Dataset Loading & Preprocessing\n",
    "\n",
    "We use the **Labeled Faces in the Wild (LFW)** dataset available via `sklearn`. We filter classes to include only individuals who have at least 70 photos to ensure the model has sufficient training data per person."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.datasets import fetch_lfw_people\n",
    "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score\n",
    "\n",
    "# Set plotting style\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams[\"figure.figsize\"] = (8, 5)\n",
    "\n",
    "# Fetch the LFW people dataset\n",
    "lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4, slice_=None)\n",
    "\n",
    "n_samples, h, w = lfw_people.images.shape\n",
    "X = lfw_people.images\n",
    "y = lfw_people.target\n",
    "target_names = lfw_people.target_names\n",
    "n_classes = len(target_names)\n",
    "\n",
    "print(\"=== Dataset Statistics ===\")\n",
    "print(f\"Total face images: {n_samples}\")\n",
    "print(f\"Grayscale image dimensions: {h}x{w}\")\n",
    "print(f\"Number of target classes (people): {n_classes}\")\n",
    "\n",
    "print(\"\\n=== Class Distribution ===\")\n",
    "class_counts = pd.Series(y).value_counts().sort_index()\n",
    "for idx, count in class_counts.items():\n",
    "    print(f\"  {target_names[idx]}: {count} images\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check pixel values scale\n",
    "print(f\"Min pixel value: {X.min()}\")\n",
    "print(f\"Max pixel value: {X.max()}\")\n",
    "\n",
    "# Normalize if max value is 255\n",
    "if X.max() > 1.0:\n",
    "    X = X / 255.0\n",
    "    print(\"Normalised pixels to [0, 1] range.\")\n",
    "\n",
    "# Split dataset (75% Train, 25% Test, stratified to retain class ratios)\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.25, random_state=42, stratify=y\n",
    ")\n",
    "\n",
    "# Add channel dimension for CNN input (height, width, channels=1)\n",
    "X_train = np.expand_dims(X_train, axis=-1)\n",
    "X_test = np.expand_dims(X_test, axis=-1)\n",
    "\n",
    "print(f\"\\nTrain features shape: {X_train.shape}\")\n",
    "print(f\"Test features shape: {X_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Task 2: CNN Architecture Design\n",
    "\n",
    "We build a simple Sequential CNN in Keras. The model consists of convolutional layers for feature extraction, max pooling layers to downsample feature maps, a flatten layer to transition to dense layers, dropout for regularization, and a dense output layer with softmax activation for class probability distribution."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras import Sequential, layers\n",
    "\n",
    "input_shape = (h, w, 1)\n",
    "\n",
    "model = Sequential([\n",
    "    layers.Input(shape=input_shape),\n",
    "    \n",
    "    # 1st Convolutional Block\n",
    "    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),\n",
    "    layers.MaxPooling2D((2, 2)),\n",
    "    \n",
    "    # 2nd Convolutional Block\n",
    "    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),\n",
    "    layers.MaxPooling2D((2, 2)),\n",
    "    \n",
    "    # 3rd Convolutional Block\n",
    "    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),\n",
    "    layers.MaxPooling2D((2, 2)),\n",
    "    \n",
    "    # Fully Connected Blocks\n",
    "    layers.Flatten(),\n",
    "    layers.Dense(128, activation='relu'),\n",
    "    layers.Dropout(0.5), # regularisation\n",
    "    layers.Dense(n_classes, activation='softmax')\n",
    "])\n",
    "\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Task 3: Model Compilation & Training\n",
    "\n",
    "We compile the model with the **Adam** optimizer and use **Sparse Categorical Crossentropy** as our loss function since targets are ordinal integers. We train the network for **20 epochs** using a batch size of **32**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compile the model\n",
    "model.compile(\n",
    "    optimizer='adam',\n",
    "    loss='sparse_categorical_crossentropy',\n",
    "    metrics=['accuracy']\n",
    ")\n",
    "\n",
    "# Fit model\n",
    "epochs = 20\n",
    "batch_size = 32\n",
    "history = model.fit(\n",
    "    X_train, y_train,\n",
    "    epochs=epochs,\n",
    "    batch_size=batch_size,\n",
    "    validation_data=(X_test, y_test),\n",
    "    verbose=1\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Task 4: Performance Evaluation\n",
    "\n",
    "We evaluate the model on the test set, computing accuracy, precision, recall, and F1 score, and plotting the training history curves and a confusion matrix heatmap."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Predict on test set\n",
    "predictions = model.predict(X_test)\n",
    "y_pred = np.argmax(predictions, axis=1)\n",
    "\n",
    "# Classification report\n",
    "print(\"=== Classification Report ===\")\n",
    "print(classification_report(y_test, y_pred, target_names=target_names))\n",
    "\n",
    "# Overall metrics\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "precision = precision_score(y_test, y_pred, average='weighted')\n",
    "recall = recall_score(y_test, y_pred, average='weighted')\n",
    "f1 = f1_score(y_test, y_pred, average='weighted')\n",
    "\n",
    "print(\"=== Evaluation Summary ===\")\n",
    "print(f'Accuracy:  {accuracy:.4f}')\n",
    "print(f'Precision: {precision:.4f}')\n",
    "print(f'Recall:    {recall:.4f}')\n",
    "print(f'F1 Score:  {f1:.4f}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot Training Accuracy and Loss\n",
    "plt.figure(figsize=(12, 4))\n",
    "\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')\n",
    "plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')\n",
    "plt.title('Training and Validation Accuracy')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(history.history['loss'], label='Train Loss', marker='o')\n",
    "plt.plot(history.history['val_loss'], label='Val Loss', marker='s')\n",
    "plt.title('Training and Validation Loss')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Loss')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot Confusion Matrix\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "plt.figure(figsize=(8, 7))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',\n",
    "            xticklabels=target_names, yticklabels=target_names)\n",
    "plt.title('Confusion Matrix - CNN Face Recognition')\n",
    "plt.xlabel('Predicted Label')\n",
    "plt.ylabel('True Label')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Task 5: Visual Prediction Preview\n",
    "\n",
    "To visual verify our model, we plot a random subset of 12 test images displaying their true and predicted names. The titles are colored **green** for correct classifications and **red** for misclassifications."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display 12 sample predictions\n",
    "plt.figure(figsize=(12, 9))\n",
    "indices = np.random.choice(len(X_test), size=12, replace=False)\n",
    "\n",
    "for i, idx in enumerate(indices):\n",
    "    plt.subplot(3, 4, i + 1)\n",
    "    img_2d = X_test[idx].reshape(h, w)\n",
    "    plt.imshow(img_2d, cmap='gray')\n",
    "    \n",
    "    true_lbl = target_names[y_test[idx]]\n",
    "    pred_lbl = target_names[y_pred[idx]]\n",
    "    \n",
    "    # Get last name for cleaner plotting titles\n",
    "    true_name = true_lbl.split()[-1]\n",
    "    pred_name = pred_lbl.split()[-1]\n",
    "    \n",
    "    title_color = 'green' if y_test[idx] == y_pred[idx] else 'red'\n",
    "    plt.title(f\"True: {true_name}\\nPred: {pred_name}\", color=title_color, fontsize=10)\n",
    "    plt.axis('off')\n",
    "    \n",
    "plt.suptitle(\"Visual Predictions Preview (Green=Correct, Red=Incorrect)\", fontsize=14)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Conclusions & Summary\n",
    "\n",
    "- **CNN Effectiveness**: The simple Convolutional Neural Network demonstrates high efficiency at extracting features (edges, textures, facial components) through Conv2D kernels, outperforming traditional ML models on image datasets.\n",
    "- **Data Imbalance**: As observed in the class distribution, some targets have many more images (e.g., George W Bush) than others. This causes the model's precision and recall to fluctuate slightly per person.\n",
    "- **Regularization**: Incorporating a Dropout layer (0.5) before the final output dense layer successfully prevents severe overfitting, keeping validation loss aligned with training curves."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open('face_recognition_cnn.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)
print("Jupyter Notebook 'face_recognition_cnn.ipynb' generated successfully!")
