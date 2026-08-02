# 🚗 Used Car Price Category Prediction - Streamlit Web Application

A production-ready, portfolio-standard Streamlit web application built with a **Notion/Vercel/Linear-inspired minimal black-and-white design system**. The app predicts used vehicle price categories using a pre-trained **Random Forest Classifier** (83.87% accuracy) trained on the CarDekho dataset.

Developer: **Rahul Simhadri**  
Model Accuracy: **83.87%**

---

## 🌟 Key Application Features

- **Minimalist Monochrome UI**: Clean Notion/Linear design language with pure `#FFFFFF` background, `#111111` typography, 12px border radii, subtle thin grey borders, and smooth fade-in transitions.
- **3-Tab Architecture**:
  - `🎯 Predict`: Interactive prediction form, instant category output card with confidence score, and model metadata overview.
  - `📈 Model Performance`: Comparative accuracy benchmarks table, monochrome Matplotlib visualizations (Feature Importance, Confusion Matrix, Model Comparison).
  - `ℹ️ About Project`: Problem statement, CarDekho dataset overview, 8-step ML pipeline workflow, and technical stack details.
- **Dynamic Brand & Model Filtering**: Interactive selection where choosing a vehicle Brand (OEM) automatically updates available Models.
- **Real-Time Input Validation**: Checks for valid ranges on kilometers driven, engine CC, power, and torque.
- **Single Centered Predict CTA**: Solid black button with loading spinner feedback.

---

## 📂 Folder Structure

```
Car_Prediction_Model/
│── app.py                              # Main Streamlit application (3-Tab structure)
│── random_forest.pkl                   # Trained Random Forest classifier model
│── scaler.pkl                          # Pre-fitted StandardScaler for numerical features
│── label_encoders.pkl                  # Fitted LabelEncoder dictionary for categorical features
│── cleaned_cars.csv                    # Processed dataset
│── requirements.txt                    # Python dependencies
│── README.md                           # Documentation & portfolio overview
│── utils/
│     ├── preprocessing.py              # Input validation, feature encoding & scaling logic
│     ├── helper.py                     # Data loader, model cache & monochrome chart plotters
```

---

## 🛠 Technologies Used

- **Python 3.10+**
- **Streamlit** (Web Application Framework)
- **Scikit-learn** (Random Forest Classifier & StandardScaler)
- **Pandas & NumPy** (Data Preprocessing & Manipulation)
- **Matplotlib & Seaborn** (Monochrome Data Visualizations)
- **Joblib** (Model Serialization & Deserialization)

---

## ⚡ Quick Start & Running Locally

### 1. Open Workspace
```bash
cd Car_Prediction_Model
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Application
```bash
streamlit run app.py
```

The application will launch at `http://localhost:8501`.

---
## Dataset

Due to GitHub's file size limitations, the cleaned dataset is not included in this repository.

You can download the original dataset from Kaggle and run the notebook to reproduce the results.

## ☁️ Deployment to Streamlit Community Cloud

1. Push this repository to **GitHub**.
2. Visit [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Click **New app** and select your GitHub repository `Car_Prediction_Model`.
4. Set Main file path to `app.py`.
5. Click **Deploy!**

---

## 📊 Model Evaluation Summary

| Model | Accuracy Score | Status |
|---|:---:|:---:|
| **Random Forest Classifier** | **83.87%** | **🏆 Best Model** |
| Decision Tree Classifier | 79.79% | Benchmark |
| Logistic Regression | 75.63% | Baseline |

---

## 📄 Attribution

Developed by **Rahul Simhadri**. Built for portfolio demonstration and deployment on Streamlit Community Cloud. Dataset sourced from CarDekho.
