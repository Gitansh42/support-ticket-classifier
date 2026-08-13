# Automated Support Ticket Classification

## 1. Project Overview

This project implements an automated text classification system for
support tickets and emails.

The system classifies incoming support tickets into one of four
categories:

- Billing
- Technical
- HR
- General

The goal is to automatically route incoming tickets while sending
low-confidence predictions for manual review.

---

## 2. Problem Statement

Support teams receive a large number of emails and tickets every day.
Manually categorizing every ticket is time-consuming and can lead to
inconsistent routing.

This project uses machine learning to automatically predict the
appropriate category for a new support ticket based on its subject
and body text.

---

## 3. Dataset

The dataset contains labeled support tickets with the following fields:

- Subject
- Body
- Category

The four target categories are:

- Billing
- Technical
- HR
- General

---

## 4. Machine Learning Pipeline

The system follows these steps:

1. Load the dataset
2. Combine subject and body text
3. Clean and normalize the text
4. Split the dataset into training and testing sets
5. Convert text into TF-IDF features
6. Train a Logistic Regression classifier
7. Evaluate the classifier
8. Generate a confusion matrix
9. Predict categories for unseen tickets
10. Calculate prediction confidence
11. Route low-confidence predictions to manual review

---

## 5. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- TF-IDF
- Logistic Regression

---

## 6. Model Evaluation

The model achieved:

**Accuracy: 87.50%**

The model correctly classified 14 out of 16 test samples.

The strongest performance was observed for:

- Billing
- Technical
- General

The HR category had some misclassification, indicating that
additional HR training examples could improve performance.

---

## 7. Unseen Ticket Testing

Five new tickets were created and passed to the trained model.

| Ticket | Predicted Category |
|---|---|
| Charged twice for one order | Billing |
| VPN connection problem | Technical |
| Salary missing | HR |
| Office working hours | General |
| Employee portal login issue | Technical |

All five sample tickets received logically appropriate predictions.

---

## 8. Confidence-Based Routing

The system also calculates prediction confidence.

To make the system safer for real-world use, predictions with confidence
below the configured threshold are sent to manual review.

This prevents uncertain predictions from being automatically routed
without human verification.

---

## 9. Results

The following files are generated automatically:

- `classification_report.txt`
- `confusion_matrix.png`
- `new_ticket_predictions.csv`

---

## 10. Limitations

The current dataset is relatively small, so the model may not
generalize perfectly to all real-world support tickets.

The HR category showed comparatively weaker performance.

A larger and more diverse dataset would improve model reliability.

---

## 11. Future Improvements

Possible improvements include:

- Increasing the training dataset
- Adding more examples for the HR category
- Testing Naive Bayes against Logistic Regression
- Hyperparameter tuning
- Adding multilingual support
- Adding a web interface using Streamlit
- Logging predictions for monitoring
- Retraining the model periodically with new labeled tickets