# 🏃 Athlete Injury Risk Predictor

## Overview
This project is a machine learning application that predicts athlete injury risk based on training load, recovery, fatigue, and performance data.

The model outputs a probability of injury and classifies the athlete into **low**, **medium**, or **high risk**, along with key contributing factors.

---

## Motivation
As a student-athlete and engineering student, I wanted to explore how data and machine learning can be used to improve performance and prevent injuries before they occur.

This project connects sports science, data analysis, and predictive modeling.

---

## Features
- Predicts injury probability using a trained ML model
- Classifies risk levels (Low / Medium / High)
- Highlights key risk factors (fatigue, recovery, workload)
- Interactive Streamlit web app
- Clean UI for real-time input and results

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (Logistic Regression + GridSearchCV)
- Streamlit (web app)
- Matplotlib, Seaborn (visualization)

---

## How It Works
1. Data is cleaned and preprocessed
2. Features are scaled using StandardScaler
3. Logistic Regression model is trained
4. Hyperparameters are optimized using GridSearchCV
5. Final model predicts injury probability
6. App displays results and risk insights

---

## Run Locally
```bash
pip install -r requirements.txt
python -m streamlit run app.py
