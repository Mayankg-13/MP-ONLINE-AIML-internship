import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Cancer Detection using Convolutional Neural Network (CNN)\n",
    "### MP ONLINE AIML Internship Project\n",
    "\n",
    "This notebook contains the complete implementation for building a **Cancer Detection** project using a Convolutional Neural Network (CNN). The objective is to identify whether a histopathology tissue image patch is normal or malignant (cancerous).\n",
    "\n",
    "To make the project fully reproducible and lightweight, we include an **H&E-stained Synthetic Cell Patch Generator** which creates simulated microscopic tissue images:\n",
    "- **Normal Cells (Class 0)**: Pink background cytoplasm with 1 to 3 regular, light-purple circular nuclei.\n",
    "- **Malignant Cells (Class 1)**: Pink background cytoplasm with 5 to 9 irregular, larger, dark-purple (hyperchromatic) nuclei, representing atypical cell structures.\n",
    "\n",
    "The assignment comprises five major tasks:\n",
    "1. **Dataset Generation & Preprocessing**: Loading synthetic stained cell images, scaling pixel values, and performing a stratified train/test split.\n",
    "2. **CNN Architecture Design**: Designing a neural network with convolutional layers, pooling, dropout, and a dense output layer with sigmoid activation.\n",
    "3. **Model Compilation & Training**: Compiling using the Adam optimizer, and training for 15 epochs.\n",
    "4. **Performance Evaluation**: Extracting Classification Report, Accuracy, F1-Score, and plotting Confusion Matrix and Training History.\n",
    "5. **Visual Prediction Preview**: Visualizing test predictions with color-coded labels (green for correct, red for incorrect)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Task 1: Dataset Generation & Preprocessing\n",
    "\n",
    "We use our custom `generate_synthetic_cell_data` function to build 1,200 total image patches (50x50 RGB)."
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
    "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score\n",
    "\n",
    "# Set plotting style\n",
    "sns.set_theme(style=\"white\")\n",
    "plt.rcParams[\"figure.figsize\"] = (8, 5)\n",
    "\n",
    "def generate_synthetic_cell_data(n_samples=1200, img_size=50):\n",
    "    X = np.zeros((n_samples, img_size, img_size, 3), dtype=np.float32)\n",
    "    y = np.zeros((n_samples,), dtype=np.int32)\n",
    "    \n",
    "    bg_color = np.array([0.94, 0.78, 0.86], dtype=np.float32) # Cytoplasm\n",
    "    normal_nucleus_color = np.array([0.31, 0.20, 0.51], dtype=np.float32) # Normal nucleus\n",
    "    cancer_nucleus_color = np.array([0.16, 0.08, 0.35], dtype=np.float32) # Malignant nucleus\n",
    "    \n",
    "    half_samples = n_samples // 2\n",
    "    for i in range(n_samples):\n",
    "        X[i] = bg_color\n",
    "        # Microscopic noise\n",
    "        noise = np.random.normal(0.0, 0.015, (img_size, img_size, 3)).astype(np.float32)\n",
    "        X[i] = np.clip(X[i] + noise, 0.0, 1.0)\n",
    "        \n",
    "        if i >= half_samples:\n",
    "            y[i] = 1 # Cancer\n",
    "            num_nuclei = np.random.randint(5, 10)\n",
    "            for _ in range(num_nuclei):\n",
    "                cx = np.random.randint(10, img_size - 10)\n",
    "                cy = np.random.randint(10, img_size - 10)\n",
    "                r_x = np.random.randint(6, 11)\n",
    "                r_y = np.random.randint(4, 9)\n",
    "                y_grid, x_grid = np.ogrid[:img_size, :img_size]\n",
    "                mask = ((x_grid - cx) / r_x)**2 + ((y_grid - cy) / r_y)**2 <= 1.0\n",
    "                X[i][mask] = cancer_nucleus_color\n",
    "        else:\n",
    "            y[i] = 0 # Normal\n",
    "            num_nuclei = np.random.randint(1, 4)\n",
    "            for _ in range(num_nuclei):\n",
    "                cx = np.random.randint(15, img_size - 15)\n",
    "                cy = np.random.randint(15, img_size - 15)\n",
    "                r = np.random.randint(4, 6)\n",
    "                y_grid, x_grid = np.ogrid[:img_size, :img_size]\n",
    "                mask = ((x_grid - cx) / r)**2 + ((y_grid - cy) / r)**2 <= 1.0\n",
    "                X[i][mask] = normal_nucleus_color\n",
    "                \n",
    "    # Box blur filter to simulate lens focus\n",
    "    for i in range(n_samples):\n",
    "        for c in range(3):\n",
    "            padded = np.pad(X[i][:, :, c], 1, mode='edge')\n",
    "            X[i][:, :, c] = (\n",
    "                padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +\n",
    "                padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +\n",
    "                padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]\n",
    "            ) / 9.0\n",
    "            \n",
    "    return X, y\n",
    "\n",
    "X, y = generate_synthetic_cell_data(1200)\n",
    "target_names = [\"Normal\", \"Malignant\"]\n",
    "\n",
    "print(f\"\\nTotal patches: {len(X)}\")\n",
    "print(f\"Normal class patches: {np.sum(y == 0)}\")\n",
    "print(f\"Malignant class patches: {np.sum(y == 1)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split into train and test sets (75% Train, 25% Test, stratified)\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.25, random_state=42, stratify=y\n",
    ")\n",
    "\n",
    "print(f\"Train features shape: {X_train.shape}\")\n",
    "print(f\"Test features shape: {X_test.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot preview of normal vs malignant patches\n",
    "plt.figure(figsize=(10, 5))\n",
    "# Normal cell example\n",
    "normal_idx = np.where(y == 0)[0][0]\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.imshow(X[normal_idx])\n",
    "plt.title(\"Class 0: Normal H&E Cell Patch\")\n",
    "plt.axis('off')\n",
    "\n",
    "# Cancer cell example\n",
    "cancer_idx = np.where(y == 1)[0][0]\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.imshow(X[cancer_idx])\n",
    "plt.title(\"Class 1: Malignant/Cancer H&E Cell Patch\")\n",
    "plt.axis('off')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Task 2: CNN Architecture Design\n",
    "\n",
    "We build a binary Sequential CNN in Keras. The model consists of three convolutional blocks to extract morphological and color shapes, followed by flattening, dense, dropout, and a final dense classification layer using sigmoid activation."
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
    "input_shape = (50, 50, 3)\n",
    "\n",
    "model = Sequential([\n",
    "    layers.Input(shape=input_shape),\n",
    "    \n",
    "    # 1st Conv Block\n",
    "    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),\n",
    "    layers.MaxPooling2D((2, 2)),\n",
    "    \n",
    "    # 2nd Conv Block\n",
    "    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),\n",
    "    layers.MaxPooling2D((2, 2)),\n",
    "    \n",
    "    # 3rd Conv Block\n",
    "    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),\n",
    "    layers.MaxPooling2D((2, 2)),\n",
    "    \n",
    "    # Fully Connected Blocks\n",
    "    layers.Flatten(),\n",
    "    layers.Dense(128, activation='relu'),\n",
    "    layers.Dropout(0.3),\n",
    "    layers.Dense(1, activation='sigmoid')\n",
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
    "We compile the model with the **Adam** optimizer and use **Binary Crossentropy** as our loss function. We train the network for **15 epochs** using a batch size of **32**."
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
    "    loss='binary_crossentropy',\n",
    "    metrics=['accuracy']\n",
    ")\n",
    "\n",
    "# Fit model\n",
    "epochs = 15\n",
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
    "y_pred = (predictions >= 0.5).astype(np.int32).flatten()\n",
    "\n",
    "# Classification report\n",
    "print(\"=== Classification Report ===\")\n",
    "print(classification_report(y_test, y_pred, target_names=target_names))\n",
    "\n",
    "# Overall metrics\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "precision = precision_score(y_test, y_pred)\n",
    "recall = recall_score(y_test, y_pred)\n",
    "f1 = f1_score(y_test, y_pred)\n",
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
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',\n",
    "            xticklabels=target_names, yticklabels=target_names)\n",
    "plt.title('Confusion Matrix - Cancer Cell Detection')\n",
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
    "To visually verify our model, we plot a random subset of 12 test images displaying their true and predicted labels with estimated probability. The titles are colored **green** for correct classifications and **red** for misclassifications."
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
    "    plt.imshow(X_test[idx])\n",
    "    \n",
    "    true_lbl = target_names[y_test[idx]]\n",
    "    pred_lbl = target_names[y_pred[idx]]\n",
    "    pred_prob = predictions[idx][0]\n",
    "    \n",
    "    title_color = 'green' if y_test[idx] == y_pred[idx] else 'red'\n",
    "    plt.title(f\"True: {true_lbl}\\nPred: {pred_lbl} ({pred_prob:.2f})\", color=title_color, fontsize=10)\n",
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
    "- **High Classifiability**: The CNN readily learns to distinguish normal cell shapes (1-3 uniform nuclei) from malignant profiles (overlapping, irregular, hyperchromatic nuclei).\n",
    "- **Staining Simulation**: Generating synthetic stained cell arrays yields clean RGB pipelines that replicate traditional histopathology classification without large disk footprint requirements.\n",
    "- **Dropout Role**: Implementing a 0.3 Dropout layer prevents cellular features from overfitting, maintaining strong generalizeability onto mock test data."
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

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
notebook_path = os.path.join(script_dir, "cancer_detection_cnn.ipynb")

with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=2)
print(f"Jupyter Notebook '{notebook_path}' generated successfully!")
