# Cancer Detection using CNN (Histopathology Cell patches)

This folder contains the project files for building a **Cancer Detection** model using a Convolutional Neural Network (CNN) trained on synthetic histopathology cell patch images.

## 📊 Project Overview

- **Dataset**: A self-contained, custom **Synthetic H&E stained Cell Image Generator** generating 1,200 total image patches ($50 \times 50 \times 3$ RGB pixels).
- **Classes**:
  - `Class 0`: Normal cells (characterized by light pink cytoplasm backgrounds and a small number of uniformly shaped, circular, light purple nuclei).
  - `Class 1`: Malignant/Cancer cells (characterized by dense, overlapping, dark/hyperchromatic nuclei with irregular boundary shapes, mimicking cellular atypia and high mitosis).
- **Goal**: Binary classification using a deep neural network to flag malignant patches.
- **Model Architecture**:
  - Input layer matching image dimensions $(50, 50, 3)$.
  - Two Convolutional blocks (Conv2D and MaxPooling2D) to extract color and morphological structures.
  - Fully Connected layers (Dense, Dropout) to map features.
  - Softmax/Sigmoid classification output layer (1 node with Sigmoid activation).

## 📁 Files in this Directory

- `main.py`: Modular Python script to generate mock images, scale variables, train the CNN, log accuracy reports, and save diagnostic plots.
- `generate_notebook.py`: Helper script to programmatically build the Jupyter Notebook.
- `cancer_detection_cnn.ipynb`: Executed Jupyter Notebook showing data statistics, cell preview, training output, and visualization curves.
- `requirements.txt`: Project dependencies list.
- `cancer_model_performance.csv`: Metrics sheet logging training scores (Accuracy, Precision, Recall, F1 Score).
- `plots/`: Output directory containing generated visual logs:
  - `training_curves.png`: Training vs. Validation Loss and Accuracy progression.
  - `confusion_matrix.png`: Heatmap highlighting true positive/negative counts.
  - `sample_predictions.png`: Visual verification grid showcasing a random 12 test cells labeled in green (correct) or red (incorrect).

## 🚀 How to Run the Script

Ensure you have your Python virtual environment activated, then install dependencies:
```bash
pip install -r requirements.txt
```

To execute the python script and train the model:
```bash
python main.py
```
This will run the synthetic cell generator, train the binary classification model, save performance stats to a CSV file, and output diagnostic plots to the `plots/` folder.
