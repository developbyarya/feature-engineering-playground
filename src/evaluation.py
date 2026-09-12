from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def evaluate_model(y_true, y_pred):
    """
    Calculates R2, MAE, and RMSE.
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {
        "R²": r2,
        "MAE": mae,
        "RMSE": rmse
    }

def plot_actual_vs_predicted(y_true, y_pred):
    """
    Creates a scatter plot of actual vs predicted values using Plotly.
    """
    fig = go.Figure()
    
    # Scatter plot of actual vs predicted
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=y_true,
        mode='markers',
        marker=dict(opacity=0.5, color='blue'),
        name='Predictions'
    ))
    
    # Perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        title='Actual vs Predicted Profit',
        xaxis_title='Predicted Profit',
        yaxis_title='Actual Profit',
        showlegend=True,
        template='plotly_white'
    )
    
    return fig
