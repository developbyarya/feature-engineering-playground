import pandas as pd
import plotly.express as px

class ExperimentTracker:
    def __init__(self):
        self.experiments = []
        
    def add_experiment(self, name, features, model_name, metrics):
        self.experiments.append({
            "Experiment": name,
            "Features": ", ".join(features),
            "Model": model_name,
            "R²": metrics["R²"],
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"]
        })
        
    def get_history_df(self):
        return pd.DataFrame(self.experiments)
        
    def plot_metric_comparison(self, metric="R²"):
        df = self.get_history_df()
        if df.empty:
            return None
            
        fig = px.bar(
            df, 
            x="R²", 
            y="Experiment", 
            orientation='h',
            title=f"Experiment Comparison ({metric})",
            text="R²",
            color="R²",
            color_continuous_scale="Viridis"
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='auto')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        return fig
