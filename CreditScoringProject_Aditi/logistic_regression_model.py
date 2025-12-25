#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Credit Scoring using Logistic Regression
Academic / Learning Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import preprocessing

import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve


# --------------------------------------------------
# Load dataset
# --------------------------------------------------
credit_df = pd.read_excel('outputAN.xlsx', sheet_name='Sheet1')


# Selected features
selected_features = [
    'Status of existing checking account',
    'Duration in month',
    'Credit history',
    'Savings account/bonds',
    'Installment rate in percentage of disposable income'
]


# Target variable cleanup
credit_df['Receive_NotReceiveCredit'] = credit_df['Receive_NotReceiveCredit'].replace(
    to_replace=2, value=0
)


# --------------------------------------------------
# Train-test split
# --------------------------------------------------
train_data, test_data = train_test_split(credit_df, test_size=0.2, random_state=42)

X_train = train_data[selected_features]
y_train = train_data['Receive_NotReceiveCredit']

X_test = test_data[selected_features]
y_test = test_data['Receive_NotReceiveCredit']


# Feature scaling
X_train_scaled = preprocessing.scale(X_train)
X_test_scaled = preprocessing.scale(X_test)


# --------------------------------------------------
# Statistical Logistic Model (for feature analysis)
# --------------------------------------------------
logit_model = sm.Logit(y_train, X_train)
logit_result = logit_model.fit()
print(logit_result.summary2())


# --------------------------------------------------
# Machine Learning Logistic Regression Model
# --------------------------------------------------
credit_model = LogisticRegression(
    penalty='l2',
    C=0.1,
    class_weight='balanced',
    max_iter=1000
)

# Train the model
credit_model.fit(X_train_scaled, y_train)


# Model parameters
print('Model Coefficients:', credit_model.coef_)
print('Model Intercept:', credit_model.intercept_)


# --------------------------------------------------
# Training accuracy
# --------------------------------------------------
train_predictions = credit_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, train_predictions)
print('Training Accuracy:', train_accuracy)


# --------------------------------------------------
# Testing accuracy
# --------------------------------------------------
test_predictions = credit_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, test_predictions)
print('Testing Accuracy:', test_accuracy)


# --------------------------------------------------
# Evaluation metrics
# --------------------------------------------------
conf_matrix = confusion_matrix(y_test, test_predictions)
print('Confusion Matrix:\n', conf_matrix)

print('\nClassification Report:\n')
print(classification_report(y_test, test_predictions))


# --------------------------------------------------
# ROC Curve
# --------------------------------------------------
roc_auc = roc_auc_score(y_test, credit_model.predict(X_test_scaled))
fpr, tpr, thresholds = roc_curve(
    y_test,
    credit_model.predict_proba(X_test_scaled)[:, 1]
)

plt.figure()
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Credit Scoring Model')
plt.legend(loc="lower right")
plt.savefig('LogisticRegression_ROC.png')
plt.show()
