# 📈 Linear Regression Salary Prediction

This project demonstrates an **end-to-end Linear Regression workflow** using Python scripts (no notebooks), following industry-style practices.

---

## 📂 Project Overview
- Exploratory Data Analysis (EDA)
- Data cleaning (missing values & outlier removal)
- Data visualization
- Simple & Multiple Linear Regression
- Model evaluation using **MSE** and **R²**

---

## 📊 Dataset
Employee salary dataset containing:
- experience
- age
- education_level
- city
- working_hours
- certifications
- salary (target)

The dataset includes missing values and outliers to simulate real-world data.

---

## 🛠️ Workflow
1. **EDA** – `eda.py`
2. **Data Cleaning** – `clean_data.py`
3. **Visualization** – `visualize.py`
4. **Simple Linear Regression** – `train1.py`
5. **Multiple Linear Regression** – `train2.py`

---

## 📈 Results
- Simple Linear Regression: R² ≈ 0.31  
- Multiple Linear Regression: R² ≈ 0.74  

---

## 🧰 Tech Stack
- Python
- Pandas, NumPy
- Matplotlib
- Scikit-learn
- VS Code

---

## ▶️ Run
```bash
pip install -r requirements.txt
python eda.py
python clean_data.py
python visualize.py
python train1.py
python train2.py