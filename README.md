# 🤖 SmartML Studio

## 📌 Overview

SmartML Studio is an interactive web application built with Streamlit that allows users to upload datasets, train machine learning models, and generate predictions without needing deep technical knowledge.

The goal is to simplify the machine learning workflow and make data analysis accessible through an intuitive interface.

---

## ⚙️ Tech Stack

* Python
* Streamlit
* Pandas
* Scikit-learn
* Matplotlib / Seaborn

---

## 🚀 Features

### 📁 File Upload

* Upload CSV datasets
* Automatic data preview
* Loading indicators for better UX

### 🤖 Machine Learning Module

* Supports **Classification** and **Regression**
* Select target variable
* Train multiple models (4 algorithms)
* Display performance metrics:

**Classification:**

* Accuracy
* Precision
* Recall
* F1-score

**Regression:**

* MAE
* MSE
* R² Score

---

### 🔮 Prediction Interface

* Input new data manually
* Select trained model
* Generate real-time predictions

---

## 📊 Project Structure

* `app.py` → Main Streamlit app
* `utils.py` → Data processing & ML functions
* `/models` → Saved trained models
* `/data` → Datasets

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎯 Objective

This project demonstrates how machine learning can be integrated into a simple web interface to support decision-making and predictive analysis.

---

## 👨‍💻 Author

Aziz Messadeg
GitHub: https://github.com/aaziz-dev
