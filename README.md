# 🧠 Data Engineering & Machine Learning Pipeline

## 📌 Overview

This project is a complete end-to-end data pipeline that covers: 
- Data Extraction
- Data Transformation
- Data Loading (ETL)
- Machine Learning Modeling
- Model Evaluation & Visualization

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

The models are evaluated using: 

<img width="500" height="1000" alt="Image" src="https://github.com/user-attachments/assets/fac2d194-6ff0-4f4f-9808-45999cbbcd72" />

<img width="500" height="1000" alt="Image" src="https://github.com/user-attachments/assets/684feea5-9f06-4a5b-ac87-8a7e23216e7e" />

---

<img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/1190c598-8659-4fd2-ae8d-243e6740610b" />
<img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/88ca9584-0290-4a93-a09b-e5f8ab70b4ad" />
<img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/d47f055a-e38e-47a5-87cc-b7f41278331b" />

---

<img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/fd588791-210f-42f9-993e-1b3c41fe0ec8" />

<img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/52d5f189-3c55-4eda-af31-978a3f1d9ad8" />

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
