# Simple Image Classification using CNN (MNIST)

This folder contains the project files for building the simplest image classification task using a Convolutional Neural Network (CNN) on the standard **MNIST handwritten digits dataset**.

## 📊 Project Overview

- **Dataset**: MNIST (60,000 training images, 10,000 testing images of 28x28 grayscale handwritten digits from 0 to 9)
- **Goal**: Build and evaluate a lightweight CNN model to identify digits with high accuracy.
- **Model Architecture**:
  - Convolutional layers for pattern extraction (edges, loops, strokes).
  - Max Pooling layers for spatial size reduction.
  - Dropout regularization to prevent overfitting.
  - Dense layers for classification mapping.
  - Softmax activation output layer (10 classes).

## 📁 Files in this Directory

- `main.py`: Modular Python script to load data, pre-process, construct, train, evaluate, and save plots and metric comparisons.
- `generate_notebook.py`: Helper script containing JSON structures to output a clean version of the Jupyter Notebook.
- `simple_image_classification.ipynb`: Fully executed Jupyter Notebook showing outputs, printed details, and inline visualization curves.
- `requirements.txt`: Project dependencies list.
- `mnist_model_performance.csv`: Saved accuracy, precision, recall, and F1 score of the trained model.
- `plots/`: Directory containing generated visual aids:
  - `training_curves.png`: Curves for accuracy and loss progression during training.
  - `confusion_matrix.png`: A heatmap plotting classification performance per class.
  - `sample_predictions.png`: Visual verification grid showcasing a random 12 test digits labeled in green (correct) or red (incorrect).

## 🚀 How to Run the Script

Ensure you have your Python virtual environment activated, then install dependencies:
```bash
pip install -r requirements.txt
```

To execute the python script and train the model:
```bash
python main.py
```
This will run the training, print the classification report, save the performance metrics to a CSV, and output visual diagnostic files in the `plots/` folder.
