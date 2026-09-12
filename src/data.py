import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath="data/Sample - Superstore.csv"):
    """
    Loads the dataset, creates base derived features, and selects required columns.
    """
    df = pd.read_csv(filepath)
    
    # Preprocessing dates
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    
    # Create derived features
    df['Shipping_Days'] = (df['Ship Date'] - df['Order Date']).dt.days
    df['Order_Year'] = df['Order Date'].dt.year
    
    # Select only the allowed features + target
    selected_columns = ['Sales', 'Quantity', 'Discount', 'Shipping_Days', 'Order_Year', 'Profit']
    
    # Filter to only keep these columns if they exist
    df = df[[col for col in selected_columns if col in df.columns]]
    
    return df

def get_train_val_split(df, target_col='Profit', test_size=0.2, random_state=42):
    """
    Splits the data into training and validation sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_val, y_train, y_val
