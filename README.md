# Project ML heart_disease

 Heart Disease Classification

This repository contains a machine learning project aimed at predicting heart disease using various Python-based libraries.

Table of Contents
- [Introduction](#introduction)
- [Problem Definition](#problem-definition)
- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)


Introduction
This project explores the use of machine learning techniques to predict whether or not a patient has heart disease based on their medical attributes. It utilizes various data analysis and machine learning libraries in Python.

Problem Definition
Given clinical parameters about a patient, can we predict whether or not they have heart disease?

Dataset
The original dataset comes from the Cleveland data from the UC Irvine Machine Learning Repository.

Evaluation
The goal is to achieve an accuracy of 95% in predicting whether or not a patient has heart disease.

Features
The dataset includes various medical attributes, such as age, sex, chest pain type, resting blood pressure, cholesterol levels, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, and others.

Requirements
To run the code in this repository, you need the following libraries:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

Usage
To use the code, run the `heart_disease_calssification.py` script. This script performs the following steps:
1. Imports necessary libraries.
2. Loads and preprocesses the dataset.
3. Splits the data into training and testing sets.
4. Trains various machine learning models.
5. Evaluates the models using different metrics.
6. Hyperparameter tuning using GridSearchCV.
7. Makes predictions and evaluates the final model.
8. Visualizes the results.

Model Training and Evaluation
The script trains multiple models including Logistic Regression, K-Nearest Neighbors, and Random Forest. It evaluates these models using:
- ROC Curve and AUC Score
- Confusion Matrix
- Classification Report
- Precision, Recall, and F1-Score

The script also includes hyperparameter tuning using GridSearchCV to find the best parameters for the Logistic Regression model.

Results
The final model is evaluated based on various metrics, and its performance is visualized using a feature importance plot.


