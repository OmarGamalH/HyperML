# 🧠 Data Engineering & Machine Learning Pipeline

## 📌 Overview

This project is a complete end-to-end data pipeline that covers: - Data
Extraction - Data Transformation - Data Loading (ETL) - Machine Learning
Modeling - Model Evaluation & Visualization

The system processes raw health datasets, prepares them for analysis,
and applies machine learning models including Logistic Regression and a
custom Neural Network.

------------------------------------------------------------------------

## ⚙️ Project Structure

    .
    ├── DE_Utilities.py     # Data Engineering utilities (ETL)
    ├── ML.py               # Machine Learning models & utilities
    ├── main.py             # Main pipeline execution
    ├── datasets/           # Raw data (input)
    ├── datasets_csv/       # Converted CSV files
    ├── bronze/             # Joined raw dataset
    ├── silver/             # Processed dataset
    ├── gold/               # Features & labels
    ├── models/             # Saved models
    ├── graphs/             # Visualizations

------------------------------------------------------------------------

## 🔄 Pipeline Workflow

### 1. Data Extraction

-   Converts `.xpt` files into `.csv`
-   Loads datasets into Pandas DataFrames

### 2. Data Integration

-   Joins multiple datasets into a unified dataset

### 3. Data Transformation

-   Feature selection & renaming
-   Handling missing values
-   Encoding categorical variables
-   Normalization
-   Class balancing

### 4. Data Loading

-   Saves processed datasets into structured layers:
    -   **Bronze** (raw merged)
    -   **Silver** (cleaned)
    -   **Gold** (features & labels)

------------------------------------------------------------------------

## 🤖 Machine Learning Models

### 1. Logistic Regression

-   Custom implementation using gradient descent
-   Compared against Scikit-learn implementation

### 2. Neural Network

-   Fully connected deep neural network
-   Supports multiple hidden layers
-   Implements forward & backward propagation from scratch

------------------------------------------------------------------------

## 📊 Evaluation

The models are evaluated using: - Accuracy Score - Confusion Matrix -
Visualization plots: - Cost vs Iterations - Model Accuracy Comparison

------------------------------------------------------------------------

## 🚀 How to Run

### 1. Install Dependencies

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn pyreadstat
```

### 2. Run the Pipeline

``` bash
python main.py
```

------------------------------------------------------------------------

## 📈 Outputs

-   Processed datasets (`silver/`, `gold/`)
-   Trained models (`models/`)
-   Performance plots (`graphs/`)

------------------------------------------------------------------------

## ⚠️ Notes & Improvements

-   Avoid data leakage by applying normalization after train-test split
-   Add more evaluation metrics (Precision, Recall, F1-score)
-   Improve scalability of dataset merging
-   Add model persistence for preprocessing steps

------------------------------------------------------------------------

## 🧩 Future Enhancements

-   Convert pipeline into a reusable package
-   Add API deployment (FastAPI / Flask)
-   Implement hyperparameter tuning
-   Add cross-validation
-   Improve logging & monitoring

------------------------------------------------------------------------

## 👨‍💻 Author

Omar Gamal Hamed
---------------------------------------------------------------------