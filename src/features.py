import numpy as np
import pandas as pd

class FeatureRegistry:
    def __init__(self):
        # Base features
        self.features = {
            "Sales": {"type": "original", "formula": "Original", "source_features": []},
            "Quantity": {"type": "original", "formula": "Original", "source_features": []},
            "Discount": {"type": "original", "formula": "Original", "source_features": []},
            "Shipping_Days": {"type": "derived", "formula": "Ship Date - Order Date", "source_features": []},
            "Order_Year": {"type": "derived", "formula": "Order Date.year", "source_features": []}
        }
    
    def add_binary_feature(self, name, feat1, operation, feat2):
        symbol = {"Addition": "+", "Subtraction": "-", "Multiplication": "×", "Division": "÷"}
        formula = f"{feat1} {symbol.get(operation, operation)} {feat2}"
        self.features[name] = {
            "type": "engineered",
            "formula": formula,
            "source_features": [feat1, feat2],
            "operation_type": "binary",
            "feat1": feat1,
            "feat2": feat2,
            "operation": operation
        }
        
    def add_unary_feature(self, name, feat1, operation):
        self.features[name] = {
            "type": "engineered",
            "formula": f"{operation}({feat1})",
            "source_features": [feat1],
            "operation_type": "unary",
            "feat1": feat1,
            "operation": operation
        }

    def add_ternary_feature(self, name, feat1, operation1, feat2, operation2, feat3):
        symbol = {"Addition": "+", "Subtraction": "-", "Multiplication": "×", "Division": "÷"}
        s1 = symbol.get(operation1, operation1)
        s2 = symbol.get(operation2, operation2)
        formula = f"({feat1} {s1} {feat2}) {s2} {feat3}"
        self.features[name] = {
            "type": "engineered",
            "formula": formula,
            "source_features": [feat1, feat2, feat3],
            "operation_type": "ternary",
            "feat1": feat1,
            "feat2": feat2,
            "feat3": feat3,
            "operation1": operation1,
            "operation2": operation2
        }

    def get_all_features(self):
        return list(self.features.keys())
        
    def get_engineered_features(self):
        return [k for k, v in self.features.items() if v["type"] == "engineered"]
        
    def get_feature_info(self, name):
        return self.features.get(name)

def apply_feature_engineering(df, feature_registry):
    """
    Applies all engineered features in the registry to the given dataframe consistently.
    """
    df_new = df.copy()
    
    for name, info in feature_registry.features.items():
        if info["type"] == "engineered":
            if info["operation_type"] == "binary":
                f1 = df_new[info["feat1"]]
                f2 = df_new[info["feat2"]]
                op = info["operation"]
                
                if op == "Addition":
                    df_new[name] = f1 + f2
                elif op == "Subtraction":
                    df_new[name] = f1 - f2
                elif op == "Multiplication":
                    df_new[name] = f1 * f2
                elif op == "Division":
                    # Handle division by zero
                    df_new[name] = np.where(f2 == 0, np.nan, f1 / f2)
                    df_new[name].fillna(0, inplace=True) # Simple imputation for learning app
                    
            elif info["operation_type"] == "unary":
                f1 = df_new[info["feat1"]]
                op = info["operation"]
                
                if op == "Square":
                    df_new[name] = np.square(f1)
                elif op == "Square Root":
                    df_new[name] = np.sqrt(np.maximum(f1, 0)) # Safe sqrt
                elif op == "Log":
                    df_new[name] = np.log1p(np.maximum(f1, 0)) # Safe log
                elif op == "Absolute Value":
                    df_new[name] = np.abs(f1)
                    
            elif info["operation_type"] == "ternary":
                f1 = df_new[info["feat1"]]
                f2 = df_new[info["feat2"]]
                f3 = df_new[info["feat3"]]
                op1 = info["operation1"]
                op2 = info["operation2"]
                
                # Step 1: f1 op1 f2
                if op1 == "Addition":
                    temp = f1 + f2
                elif op1 == "Subtraction":
                    temp = f1 - f2
                elif op1 == "Multiplication":
                    temp = f1 * f2
                elif op1 == "Division":
                    temp = np.where(f2 == 0, np.nan, f1 / f2)
                    temp = pd.Series(temp).fillna(0)
                else:
                    temp = f1
                    
                # Step 2: temp op2 f3
                if op2 == "Addition":
                    df_new[name] = temp + f3
                elif op2 == "Subtraction":
                    df_new[name] = temp - f3
                elif op2 == "Multiplication":
                    df_new[name] = temp * f3
                elif op2 == "Division":
                    res = np.where(f3 == 0, np.nan, temp / f3)
                    df_new[name] = pd.Series(res).fillna(0)
                else:
                    df_new[name] = temp
                    
    return df_new
