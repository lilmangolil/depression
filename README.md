# Geriatric Depression Screening Tool

A machine learning-based screening tool for geriatric depression, designed to assist primary healthcare workers in assessing depression risk in elderly populations through a user-friendly web interface.

## Overview

This project implements a depression risk screening tool specifically tailored for elderly individuals. The tool utilizes machine learning models to analyze 13 key feature variables and provide real-time risk assessment results (high/low risk). The system consists of a frontend interface built with HTML/CSS and a backend powered by Python's Flask framework, ensuring ease of use for healthcare professionals in primary care settings.

## Project Structure

├── code/│ ├── templates/│ │ └── index.html # Frontend interface for user input and result display│ ├── Data_Concat.py # Data concatenation and merging functions│ ├── Data_Preprocessing.py # Data preprocessing and feature engineering│ ├── Model_Build.py # Model training and evaluation pipeline│ └── app.py # Flask backend server handling calculations└── README.md


## File Descriptions

### 1. `templates/index.html`
- HTML/CSS interface for user interaction
- Provides input fields for 13 feature variables including:
  - Demographic information (gender, education level)
  - Health conditions (hypertension, diabetes, asthma, sleep disorders)
  - Lifestyle factors (alcohol use, sedentary behavior)
  - Biological markers (BMI, albumin, neutrophils)
  - Other metrics (HEI-2020 score, PIR)
- Displays risk assessment results (high/low risk) with corresponding recommendations
- Responsive design optimized for different screen sizes

### 2. `Data_Concat.py`
- Contains functions for concatenating and merging multi-year datasets
- `data_concat()`: Main function that processes year-specific file paths
- `merge_by_path()`: Helper function that merges datasets using a common identifier (SEQN)
- Handles outer joins to preserve maximum data while combining multiple sources

### 3. `Data_Preprocessing.py`
- Implements data preprocessing pipeline for machine learning
- Uses `ColumnTransformer` to handle different preprocessing for categorical and numeric features:
  - One-hot encoding for categorical variables
  - Standard scaling for continuous variables
- Processes training and test sets separately to prevent data leakage
- Returns preprocessed data ready for model training/evaluation

### 4. `Model_Build.py`
- Provides model training, cross-validation, and evaluation framework
- Supports multiple models through a dictionary-based input structure
- Implements comprehensive evaluation metrics:
  - F1-score, precision, recall (macro-averaged)
  - Accuracy and ROC-AUC
  - Specificity (custom metric)
- Uses 10-fold stratified cross-validation for robust model assessment
- Returns training and test performance results as DataFrames

### 5. `app.py`
- Flask web application handling backend logic
- Loads pre-trained model (`depression_model.pkl`) for predictions
- Processes user input from the frontend form
- Converts categorical inputs to model-compatible numeric values
- Generates risk assessment results and redirects to display page
- Handles basic error checking for model loading

## Disclaimer

This tool is intended for screening purposes only and should not be used as a substitute for professional medical diagnosis. All results should be interpreted by qualified healthcare professionals.

## Important Usage Notice  
The code provided in this repository is intended for **methodological reference only** and constitutes a subset of the complete analysis pipeline used in the associated study.  

### Key Limitations:  
1.  **Exemplary Nature**: These modules demonstrate core algorithmic approaches with generalized parameterization. Function/variable naming conventions are standardized for clarity and may differ from the production implementation.  
2.  **Non-reproducibility**: This partial codebase alone cannot regenerate published results. Execution requires integration with proprietary preprocessing frameworks, institutional data governance systems, and licensed third-party dependencies not included here.  
3.  **No Support Guarantee**: The code is provided "AS IS" without warranty of completeness or functionality. No user support, bug fixes, or compatibility updates will be provided.  

The pre-trained model file (`depression_model.pkl`) is not included in this repository. For access to the complete model and additional resources, please contact the corresponding author after the associated research paper is published. Full analysis scripts will be released post-publication in compliance with journal policy. Researchers may request access via the corresponding author per the manuscript's Data Availability statement.  