# Feature Engineering Playground

**Feature Engineering Playground** is a web-based educational application built with Streamlit for Data Science courses. It provides a no-code, interactive learning activity for students to understand the impact of feature engineering on machine learning models.

The primary concept is:
**Select Features → Create/Transform Features → Select Features for Model → Train → Evaluate → Compare**

## Learning Objectives

- Understand feature engineering concepts without writing code.
- Create new features using mathematical operations and unary transformations.
- Compare model performance (R², MAE, RMSE) before and after engineering features.
- Learn that not all features improve performance, and good feature engineering requires understanding the data.

## Installation

1. Make sure you have Python 3.11+ installed.
2. Clone this repository or download the files.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

To start the Streamlit application, navigate to the project directory and run:

```bash
streamlit run app.py
```

Open the provided local URL (usually `http://localhost:8501`) in your web browser.

## Dataset

The application uses the `Sample - Superstore.csv` dataset. For educational purposes, it is simplified to the following features:
- `Sales`: Total sales value
- `Quantity`: Number of products sold
- `Discount`: Discount applied
- `Shipping_Days`: Days between Order Date and Ship Date (Derived)
- `Order_Year`: Year of the transaction (Derived)

**Target Variable**: `Profit` (Regression Problem)

*Note: To prevent target leakage, `Profit` cannot be selected as an input feature.*

## Available Models

- **Linear Regression**: A simple baseline to observe linear relationships. Data is automatically scaled using `StandardScaler`.
- **Decision Tree Regressor**: A simple non-linear model.
- **Random Forest Regressor**: A more powerful ensemble model.

## Feature Engineering Interface

Students can create new features in the "Feature Builder" section:
- **Combine 2 Features**: Apply addition, subtraction, multiplication, or division to two existing features (e.g., `Sales / Quantity`).
- **Transform 1 Feature**: Apply transformations like Square, Square Root, Log, or Absolute Value to a single feature.

Division by zero and invalid logarithm inputs are handled automatically.

## Evaluation Metrics

- **R²**: Proportion of variance explained by the model (Higher is better).
- **MAE**: Mean Absolute Error (Lower is better).
- **RMSE**: Root Mean Squared Error (Lower is better).
