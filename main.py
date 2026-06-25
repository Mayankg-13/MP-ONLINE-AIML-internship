import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def load_data(filepath="adult.csv"):
    """Load the dataset and do initial inspection."""
    print("--- Task 1: Dataset Understanding ---")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file {filepath} not found.")
    
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print("\nColumns and Data Types:")
    print(df.dtypes)
    
    print("\nClass distribution in target variable ('income'):")
    class_counts = df['income'].value_counts()
    for val, count in class_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {val}: {count} ({pct:.2f}%)")
        
    return df

def clean_data(df):
    """Clean the data (handling missing values, whitespaces, and duplicates)."""
    print("\n--- Task 2: Data Cleaning ---")
    df_cleaned = df.copy()
    
    # 1. Strip whitespace from all string columns
    string_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in string_cols:
        df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
        
    # 2. Check for missing values (represented by '?')
    print("Checking for missing values represented as '?'...")
    missing_counts = {}
    for col in df_cleaned.columns:
        q_count = (df_cleaned[col] == '?').sum()
        if q_count > 0:
            missing_counts[col] = q_count
            print(f"  Column '{col}': {q_count} missing values ('?')")
            
    if not missing_counts:
        print("  No '?' missing values found.")
    else:
        # Option A: Replace '?' with 'Unknown'
        print("Handling missing values: Replacing '?' with 'Unknown'")
        for col in missing_counts.keys():
            df_cleaned[col] = df_cleaned[col].replace('?', 'Unknown')
            
    # 3. Check and drop duplicate rows
    duplicate_count = df_cleaned.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")
    if duplicate_count > 0:
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"  Dropped duplicates. New shape: {df_cleaned.shape}")
        
    return df_cleaned

def preprocess_and_split(df):
    """Handle feature engineering: scale continuous columns, encode categories, split data."""
    print("\n--- Task 3: Feature Engineering ---")
    
    # Drop redundant column 'education' because 'education.num' contains the same ordering
    print("Dropping redundant column 'education' (using 'education.num' instead).")
    df_fe = df.drop(columns=['education'])
    
    # Separate features and target
    X = df_fe.drop(columns=['income'])
    y = df_fe['income']
    
    # Encode target variable: '<=50K' -> 0, '>50K' -> 1
    # Check for both '<=50K' and '<=50K.' (as sometimes in test datasets there is a period)
    y = y.apply(lambda x: 1 if '>' in str(x) else 0)
    print("Target variable encoded: 0 for <=50K, 1 for >50K.")
    
    # Identify numerical and categorical features
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    
    # Preprocessor for numerical features (Standard Scaling) and categorical features (One-Hot Encoding)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # Train-test split (80-20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor

def build_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """Build and evaluate the 5 classification models."""
    print("\n--- Task 4 & 5: Model Building & Performance Evaluation ---")
    
    # Make directory for plots if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Define the 5 classification algorithms
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "SVM": SVC(probability=True, random_state=42)
    }
    
    results = []
    
    # Set up matplotlib figure for ROC curves
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Build pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        
        # Probabilities for ROC-AUC
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_prob = pipeline.decision_function(X_test)
        else:
            y_prob = y_pred
            
        # Calculate performance metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        print(f"Evaluation of {name}:")
        print(classification_report(y_test, y_pred))
        
        results.append({
            "Algorithm": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": roc_auc
        })
        
        # Save Confusion Matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['<=50K', '>50K'], 
                    yticklabels=['<=50K', '>50K'])
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f"plots/confusion_matrix_{name.lower().replace(' ', '_')}.png")
        plt.close()
        
        # Add to the joint ROC Curve plot
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(1) # Select the joint ROC plot
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")
        
    # Finalize and save joint ROC Curve plot
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/roc_curves_comparison.png")
    plt.close()
    
    # Create final comparison DataFrame
    results_df = pd.DataFrame(results)
    print("\n--- Final Performance Evaluation Comparison Table ---")
    # Format for clean console printing
    print(results_df.to_string(index=False))
    
    # Save metrics table to CSV
    results_df.to_csv("model_comparison_results.csv", index=False)
    print("\nSaved evaluation table to 'model_comparison_results.csv'")
    print("Saved all visual plots to 'plots/' directory")
    
    return results_df

def main():
    # Load
    df = load_data()
    
    # Clean
    df_cleaned = clean_data(df)
    
    # Feature Engineer and Preprocess
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df_cleaned)
    
    # Build models and evaluate
    build_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
    
    print("\nAssignment processing completed successfully!")

if __name__ == "__main__":
    main()
