# Stroke Risk ML Model Comparison

This project analyzes the healthcare-dataset-stroke-data.csv dataset to predict the likelihood of a stroke using several machine learning algorithms, and compares their performance side by side.

## What it does

Cleans and preprocesses the healthcare dataset (handles missing values, encodes categorical features). Trains and evaluates multiple classification models: Logistic Regression, Decision Tree, Random Forest, SVM, and K-Nearest Neighbors. Compares models using accuracy, confusion matrices, and ROC curves. Generates visualizations including a correlation matrix, feature importance chart, missing value overview, and a merged PDF report.

## Tech stack

Python, pandas, numpy, scikit-learn, matplotlib, seaborn, imbalanced-learn, and reportlab for the PDF report.

## Getting started

Clone the repository, install dependencies with pip install -r requirements.txt, then run the main script with python stroke_prediction.py. Check the analysis folder for generated charts, confusion matrices, ROC curves, and the merged PDF report.

## Results

Model comparison charts, confusion matrices, and ROC curves for each algorithm are available in the analysis folder, along with a summary in model_results.txt.
