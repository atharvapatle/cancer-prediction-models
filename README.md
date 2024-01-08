# Gene Expression Classification

## Introduction

This repository contains code for the gene expression classification project, which aims to explore the use of various machine learning models for classifying gene expression data. The project involves methodologies, data preprocessing, model training, and results analysis.

## Prerequisites

Before running the code, ensure you have the following Python libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- keras

You can install these libraries using the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost keras
```

## Dataset

The gene expression data is provided in CSV format, including training and test sets with corresponding labels for classification.

## Data Preprocessing

Data preprocessing involves several key steps to prepare the dataset for model training:

1. **Handling Missing Values:** Any missing values in the dataset are addressed appropriately.
2. **Label Encoding:** Categorical labels are encoded into numerical values.
3. **Feature Scaling:** Features are scaled to bring them to a similar scale.
4. **Principal Component Analysis (PCA):** Dimensionality reduction using PCA is applied to the dataset.

## Model Training

The project includes the training of various machine learning models, each with its specific focus and methodology:

### Support Vector Machine (SVM)

- Hyperparameter optimization.
- Visualization of decision boundaries.
- Classification report and confusion matrix.

### Random Forest

- Hyperparameter tuning using GridSearchCV.
- Feature importance analysis.
- Learning curve and ROC curve.

### Neural Network

- Hyperparameter search.
- Model training with early stopping.
- Confusion matrix.
- Validation loss and accuracy analysis.

## Results

The accuracy of the trained models is compared and displayed in a bar plot. The maximum accuracy achieved is highlighted for easy interpretation.

For in-depth details and code implementation, please refer to the provided Jupyter Notebook and code files in this project.

**Gene Expression Classification**
