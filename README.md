# HyperML

## Overview

This project focuses on predicting hypertension using a machine learning
pipeline built from scratch and with Scikit-learn for comparison. It
demonstrates a complete data engineering and modeling workflow,
including data extraction, transformation, loading (ETL), model
training, evaluation, and visualization.

------------------------------------------------------------------------

## Dataset Description

The dataset contains the following features:

-   **gender**: Binary encoded gender (0 = Female, 1 = Male)
-   **age_at_years**: Age of the individual
-   **Body_Mass_Index (BMI)**: Body mass index
-   **Frequency_of_moderate_LTPA**: Frequency of moderate leisure-time
    physical activity
-   **had_diabetes**: Diabetes status (0 = No, 1 = Yes)
-   \*\*Sodium\_(mg)\_perday\*\*: Daily sodium intake

### Target Variable

-   **target**: Hypertension status (0 = No, 1 = Yes)

------------------------------------------------------------------------

## Project Structure

    ├── datasets/              # Raw XPT datasets
    ├── datasets_csv/          # Converted CSV datasets
    ├── bronze/                # Joined raw data
    ├── silver/                # Processed dataset
    ├── gold/                  # Final features (X) and labels (Y)
    ├── graphs/                # Generated plots
    ├── models/                # Saved models
    ├── DE_Utilities.py        # Data engineering utilities (ETL)
    ├── ML.py                  # Custom ML implementation
    ├── main.py                # Pipeline entry point
    └── README.md

------------------------------------------------------------------------

## Pipeline Workflow

### 1. Data Extraction

-   Converts `.xpt` files into `.csv`
-   Merges datasets using a common key

### 2. Data Transformation

-   Selects relevant features
-   Handles missing values
-   Encodes categorical variables
-   Normalizes numerical features
-   Balances dataset classes

### 3. Data Loading

-   Stores processed datasets into structured layers:
    -   Bronze (raw merged)
    -   Silver (cleaned)
    -   Gold (final ML-ready)

### 4. Model Training

Two models are trained: 
- Custom Logistic Regression (implemented from scratch)
- Scikit-learn Logistic Regression

### 5. Evaluation

-   Accuracy comparison between both models
-   Visualization:
    -   Cost vs Iterations
    -   Model Accuracy Comparison

<img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/8b9ea230-08ae-4dcc-a134-2f10e237823b" />
<img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/cbab3a16-f3f3-4a41-9e2d-65ef1c07b625" />
<img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/4ecb74c1-4cc7-4bc6-8756-d602dafd057b" />

------------------------------------------------------------------------

## How to Run

### Requirements

Install dependencies:

    pip install numpy pandas matplotlib scikit-learn pyreadstat

### Run the Pipeline

    python main.py

------------------------------------------------------------------------

## Outputs

-   **Processed Data**:
    -   `silver/processed.csv`
    -   `gold/X.csv`
    -   `gold/Y.csv`
-   **Models**:
    -   `models/sklearn_model.pkl`
    -   `models/my_model.pkl`
-   **Visualizations**:
    -   `graphs/costs_figure.png`
    -   `graphs/accuracy_figure.png`

------------------------------------------------------------------------

## Key Features

-   End-to-end ML pipeline
-   Custom implementation of logistic regression
-   Data balancing and normalization
-   Modular ETL design
-   Model comparison and visualization

------------------------------------------------------------------------

## Future Improvements

-   Hyperparameter tuning with cross-validation
-   Additional evaluation metrics (Precision, Recall, F1-score)
-   Deployment as an API
-   Feature importance analysis

------------------------------------------------------------------------

## Author

Omar Gamal

------------------------------------------------------------------------

## License

This project is for educational purposes.
