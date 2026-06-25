# Adult Census Income Classification Project
### MP ONLINE AIML Internship Assignment

This repository contains the complete implementation of the **Adult Census Income Classification** assignment. The objective is to predict whether an individual's annual income exceeds **$50,000** based on demographic and employment-related features.

---

## 📋 Assignment Overview

The project is structured into 5 main tasks:
1. **Task 1: Dataset Understanding (10 Marks)** - Inspecting dataset schema, shape, data types, and checking class balance/imbalance in the target variable (`income`).
2. **Task 2: Data Cleaning (20 Marks)** - White-space trimming, identifying and handling missing values represented by `?`, and removing duplicates.
3. **Task 3: Feature Engineering (15 Marks)** - Dropping redundant fields, target variable mapping, standard scaling for numeric features, one-hot encoding for categorical attributes, and stratified splitting.
4. **Task 4: Model Building (30 Marks)** - Training 5 classification algorithms: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM).
5. **Task 5: Performance Evaluation (15 Marks)** - Evaluation using Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics, including confusion matrix plotting and ROC curve visualization.

---

## 🗂️ Dataset Description

The dataset used is the **Adult Census Income Dataset**, containing **32,561 rows** and **15 columns**:
* **Continuous Features**: `age`, `fnlwgt`, `education.num`, `capital.gain`, `capital.loss`, `hours.per.week`.
* **Categorical Features**: `workclass`, `education`, `marital.status`, `occupation`, `relationship`, `race`, `sex`, `native.country`.
* **Target Variable**: `income` (binary classification: `<=50K` or `>50K`).

---

## 🚀 Final Performance Evaluation Table

The performance metrics for all five classification algorithms on the test set are summarized below:

| Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 85.14% | 0.7346 | 0.6001 | 0.6606 | **0.9019** |
| **Decision Tree** | 84.77% | 0.7095 | **0.6231** | **0.6635** | 0.8846 |
| **Random Forest** | 85.08% | 0.7267 | 0.6103 | 0.6634 | 0.8958 |
| **KNN** | 82.74% | 0.6609 | 0.5829 | 0.6195 | 0.8517 |
| **SVM** | **85.22%** | **0.7475** | 0.5835 | 0.6554 | 0.8960 |

### 🔍 Key Insights & Conclusions
* **Overall Best Classifier**: **Support Vector Machine (SVM)** achieved the highest overall **Accuracy (85.22%)** and **Precision (74.75%)**.
* **F1 Score Winner**: **Decision Tree** (0.6635) and **Random Forest** (0.6634) achieved the highest F1-scores, striking the best balance between precision and recall.
* **ROC-AUC Winner**: **Logistic Regression** achieved the highest ROC-AUC value (**0.9019**), indicating excellent class-separation capability.
* **Class Imbalance Impact**: The dataset is heavily imbalanced (~76% make `<=50K` and ~24% make `>50K`). For this reason, simple Accuracy is not a sufficient metric; **F1-Score** and **ROC-AUC** are critical metrics to evaluate how well models classify the minority high-income group.
* **Redundancy Reduction**: Dropping the `education` column in favor of `education.num` (which represents the exact same information in ordinal numeric form) reduced dimensionality without any loss in predictive power.

---

## 📈 Visualizations

The pipeline automatically generates and saves visual figures in the `plots/` directory:
1. **ROC Curves Comparison** (`plots/roc_curves_comparison.png`): Shows the Receiver Operating Characteristic curves for all 5 models together, comparing true positive vs. false positive rates.
2. **Confusion Matrices**: Shows true vs. predicted counts for each model:
   * `plots/confusion_matrix_logistic_regression.png`
   * `plots/confusion_matrix_decision_tree.png`
   * `plots/confusion_matrix_random_forest.png`
   * `plots/confusion_matrix_knn.png`
   * `plots/confusion_matrix_svm.png`

---

## 🛠️ Installation & Setup Instructions

To run this project locally, follow these steps:

### 1. Prerequisites
Ensure you have **Python 3.10+** installed.

### 2. Create and Activate a Virtual Environment
Navigate to the project root directory and run:

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline Script
To run the automated script that cleans the data, trains the models, saves the comparison table, and generates the plots, execute:
```bash
python main.py
```

### 5. Open the Jupyter Notebook
To run the notebook with step-by-step documentation, explanations, and inline visualizations:
```bash
jupyter notebook adult_census_income_analysis.ipynb
```

---

## 📦 Project Structure

```
├── adult.csv                                # Raw Dataset
├── requirements.txt                         # Dependencies
├── .gitignore                               # Files to ignore in Git
├── main.py                                  # Modular Python ML pipeline script
├── generate_notebook.py                     # Helper script to create Jupyter notebook
├── adult_census_income_analysis.ipynb       # Complete step-by-step Jupyter Notebook
├── model_comparison_results.csv             # Saved final comparison metrics table
├── README.md                                # Project Documentation (this file)
└── plots/                                   # Visualizations
    ├── roc_curves_comparison.png
    ├── confusion_matrix_logistic_regression.png
    ├── confusion_matrix_decision_tree.png
    ├── confusion_matrix_random_forest.png
    ├── confusion_matrix_knn.png
    └── confusion_matrix_svm.png
```

---

## 🐙 How to Submit / Upload to GitHub

To upload this workspace to your personal, public GitHub repository, follow these instructions:

1. **Log in to GitHub** and click the **New** repository button.
2. Set the repository name (e.g. `mp-online-census-classification`), write a brief description, set the visibility to **Public**, and do **NOT** add a README, gitignore, or license (since they are already created here). Click **Create repository**.
3. Copy the URL of your new repository (e.g. `https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git`).
4. In your local terminal, run the following commands to commit files and push them to your repository:
   ```bash
   git add .
   git commit -m "Initial commit: Complete Adult Census Income Classification assignment"
   git branch -M main
   git remote add origin YOUR_GITHUB_REPOSITORY_URL
   git push -u origin main
   ```
5. Your project will now be live on GitHub and ready for submission!
