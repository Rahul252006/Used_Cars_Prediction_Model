# 🚗 Used Car Price Category Prediction

A machine learning project that classifies used cars into different price categories using vehicle specifications from the CarDekho dataset.

---

## 📌 Business Problem

Used car dealerships handle thousands of vehicles with different specifications. Manually categorizing vehicles based on price is time-consuming and inconsistent.

This project builds a machine learning classification model that automatically predicts the price category of a used car based on its features.

---

## 🎯 Objective

- Perform data cleaning and preprocessing on a real-world dataset.
- Handle missing values and inconsistent data.
- Perform Exploratory Data Analysis (EDA).
- Engineer and prepare features for machine learning.
- Train multiple classification models.
- Compare model performance.
- Identify the most important factors affecting vehicle price categories.

---

## 📂 Dataset

**Source:** CarDekho Used Cars Dataset

- 13,577 Records
- 140 Original Features
- Real-world uncleaned automotive dataset

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 📊 Data Preprocessing

- Removed duplicate and metadata columns
- Handled missing values
- Converted text-based numerical columns
- Created target variable (Price Category)
- Encoded categorical features
- Standardized numerical features

---

## 📈 Exploratory Data Analysis

Performed:

- Dataset Profiling
- Missing Value Analysis
- Feature Importance Analysis
- Correlation Analysis
- Distribution Analysis
- Category-wise Analysis

---

## 🤖 Machine Learning Models

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

## 📌 Model Performance

| Model | Accuracy |
|--------|---------:|
| Random Forest | **83.87%** |
| Decision Tree | **79.79%** |
| Logistic Regression | **75.63%** |

---

## 📊 Feature Importance

Top features influencing the prediction include:

- Model Year
- Maximum Power
- Alloy Wheel Size
- Width
- Kilometers Driven
- Gear Box
- Wheel Base
- Maximum Torque

---

## 📁 Project Structure

```
Used-Car-Price-Category-Prediction/
│
├── Used_Car_Price_Category_Prediction.ipynb
├── cleaned_cars.csv
├── random_forest.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── screenshots/
```

---

## 🚀 Future Improvements

- Hyperparameter Tuning
- Streamlit Web Application
- Model Deployment
- Feature Selection Optimization
- API Integration

---

## 📄 License

This project is created for educational purposes and portfolio demonstration.
