from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def get_model(model_name):
    """
    Returns the scikit-learn model pipeline based on the name.
    """
    # Base imputers to handle NaNs (e.g. from division by zero if not fully filled, or missing data)
    imputer = SimpleImputer(strategy='median')
    
    if model_name == "Linear Regression":
        # Linear Regression needs scaling
        return Pipeline([
            ('imputer', imputer),
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])
    elif model_name == "Decision Tree Regressor":
        return Pipeline([
            ('imputer', imputer),
            ('model', DecisionTreeRegressor(max_depth=5, random_state=42))
        ])
    elif model_name == "Random Forest Regressor":
        return Pipeline([
            ('imputer', imputer),
            ('model', RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1))
        ])
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def train_model(model, X_train, y_train):
    """
    Trains the model on the training data.
    """
    model.fit(X_train, y_train)
    return model

def predict_model(model, X):
    """
    Predicts using the trained model.
    """
    return model.predict(X)
