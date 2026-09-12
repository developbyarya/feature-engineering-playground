import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import re

from src.data import load_and_preprocess_data, get_train_val_split
from src.features import FeatureRegistry, apply_feature_engineering
from src.models import get_model, train_model, predict_model
from src.evaluation import evaluate_model, plot_actual_vs_predicted
from src.experiments import ExperimentTracker
from src.pdf_report import generate_pdf_report

# Page config
st.set_page_config(
    page_title="Feature Engineering Playground",
    page_icon="🧪",
    layout="wide"
)

# Initialize Session State
if "feature_registry" not in st.session_state:
    st.session_state.feature_registry = FeatureRegistry()
if "experiment_tracker" not in st.session_state:
    st.session_state.experiment_tracker = ExperimentTracker()
if "baseline_metrics" not in st.session_state:
    st.session_state.baseline_metrics = None
if "df" not in st.session_state:
    try:
        st.session_state.df = load_and_preprocess_data("data/Sample - Superstore.csv")
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure data/Sample - Superstore.csv exists.")
        st.stop()
if "exp_count" not in st.session_state:
    st.session_state.exp_count = 1
if "group_information" not in st.session_state:
    st.session_state.group_information = {"group_name": "", "members": []}
if "latest_metrics" not in st.session_state:
    st.session_state.latest_metrics = None
if "latest_selected_features" not in st.session_state:
    st.session_state.latest_selected_features = []

HARDCODED_MODEL = "Decision Tree Regressor"

def run_experiment(name, features_to_use):
    # 1. Apply features to original dataframe
    engineered_df = apply_feature_engineering(st.session_state.df, st.session_state.feature_registry)
    
    # 2. Split data
    X_train, X_val, y_train, y_val = get_train_val_split(engineered_df, target_col='Profit', test_size=0.2, random_state=42)
    
    # 3. Filter selected features
    X_train_sel = X_train[features_to_use]
    X_val_sel = X_val[features_to_use]
    
    # 4. Train Model
    model = get_model(HARDCODED_MODEL)
    model = train_model(model, X_train_sel, y_train)
    
    # 5. Predict & Evaluate
    y_pred = predict_model(model, X_val_sel)
    metrics = evaluate_model(y_val, y_pred)
    
    return metrics, y_val, y_pred

# Main Title
st.title("🧪 FEATURE ENGINEERING PLAYGROUND")
st.markdown("Learn how better features can improve ML")

st.divider()

# --- Section: GROUP INFORMATION ---
st.header("👥 GROUP INFORMATION")

with st.expander("Group + Student Names/NIMs", expanded=True):
    st.markdown("**Members**")
    members_data = []
    
    # Save the current state of members internally to prefill
    current_members = st.session_state.group_information.get("members", [])
    for i in range(4):
        default_name = current_members[i]["name"] if i < len(current_members) else ""
        default_nim = current_members[i]["nim"] if i < len(current_members) else ""
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input(f"Student {i+1} Name", value=default_name, key=f"std_name_{i}")
        with col2:
            nim = st.text_input(f"Student {i+1} NIM", value=default_nim, key=f"std_nim_{i}")
            
        if name or nim:
            members_data.append({"name": name, "nim": nim})
            
    # Update session state on change
    st.session_state.group_information = {
        "members": members_data
    }

st.divider()

# --- Auto Calculate Baseline ---
original_features = ["Sales", "Quantity", "Discount", "Shipping_Days", "Order_Year"]
if st.session_state.baseline_metrics is None:
    bl_metrics, _, _ = run_experiment("Baseline", original_features)
    st.session_state.baseline_metrics = bl_metrics
    st.session_state.experiment_tracker.add_experiment("Baseline", original_features, HARDCODED_MODEL, bl_metrics)

# --- Section: DATA EXPLORER ---
st.header("📊 DATA EXPLORER")

tab_overview, tab_dist, tab_rel, tab_stats = st.tabs(["Overview", "Distributions", "Relationships", "Statistics"])

with tab_overview:
    st.markdown("### Dataset Overview")
    st.markdown(f"**Rows**: {len(st.session_state.df):,}\n\n**Input Features**: 5\n\n**Target**: Profit\n\n**Problem Type**: Regression")
    st.markdown("### Data Preview")
    st.dataframe(st.session_state.df.head(10))

with tab_dist:
    num_cols = ["Sales", "Quantity", "Discount", "Shipping_Days", "Order_Year", "Profit"]
    dist_feat = st.selectbox("Select feature", options=num_cols, key="dist_feat_sel")
    if dist_feat:
        fig_dist = px.histogram(st.session_state.df, x=dist_feat, title=f"Distribution of {dist_feat}")
        st.plotly_chart(fig_dist, use_container_width=True)

with tab_rel:
    col_rel1, col_rel2 = st.columns(2)
    with col_rel1:
        st.markdown("### Feature vs Profit")
        x_feat = st.selectbox("X-axis:", options=["Sales", "Quantity", "Discount", "Shipping_Days", "Order_Year"], key="x_feat_sel")
        st.markdown("**Y-axis**: Profit")
        if x_feat:
            fig_scatter = px.scatter(st.session_state.df, x=x_feat, y="Profit", title=f"{x_feat} vs Profit", opacity=0.5)
            st.plotly_chart(fig_scatter, use_container_width=True)
    with col_rel2:
        st.markdown("### Correlation Heatmap")
        corr = st.session_state.df[num_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

with tab_stats:
    st.markdown("### Summary Statistics")
    st.dataframe(st.session_state.df[num_cols].describe().T[['mean', '50%', 'min', 'max']].rename(columns={'50%': 'median'}))

st.divider()

# --- Section: BASELINE ---
st.header("📈 BASELINE")
b_col1, b_col2, b_col3 = st.columns(3)
b_col1.metric("R²", f"{st.session_state.baseline_metrics['R²']:.4f}")
b_col2.metric("MAE", f"{st.session_state.baseline_metrics['MAE']:.2f}")
b_col3.metric("RMSE", f"{st.session_state.baseline_metrics['RMSE']:.2f}")

st.divider()

# --- Section: FEATURE ENGINEERING ---
st.header("🧩 FEATURE ENGINEERING")
st.info("Create a new representation of your data that may help the model discover useful patterns.")

col_feat_exist, col_feat_build = st.columns(2)

with col_feat_exist:
    st.subheader("Available Features")
    st.markdown("**Original Features**")
    for feat in ["Sales", "Quantity", "Discount", "Shipping_Days", "Order_Year"]:
        st.markdown(f"• {feat}")
    
    st.markdown("**Target**")
    st.markdown("• Profit")
        
    st.markdown("**Your Engineered Features**")
    eng_features = st.session_state.feature_registry.get_engineered_features()
    if not eng_features:
        st.markdown("*None yet.*")
    else:
        for feat in eng_features:
            info = st.session_state.feature_registry.get_feature_info(feat)
            st.markdown(f"**{feat}**\n\n*Formula: {info['formula']}*")

with col_feat_build:
    st.subheader("Create New Feature")
    tab1, tab2, tab3 = st.tabs(["Combine 2 Features", "Combine 3 Features", "Transform 1 Feature"])
    
    all_current_features = st.session_state.feature_registry.get_all_features()
    
    with tab1:
        col_f1, col_op, col_f2 = st.columns(3)
        with col_f1:
            f1 = st.selectbox("Feature A", all_current_features, key="bin_f1")
        with col_op:
            op = st.selectbox("Operation", ["Addition", "Subtraction", "Multiplication", "Division"], key="bin_op")
        with col_f2:
            f2 = st.selectbox("Feature B", all_current_features, key="bin_f2")
            
        new_feat_name = st.text_input("Feature Name", key="bin_name")
            
        if st.button("CREATE FEATURE", key="bin_btn"):
            if not new_feat_name:
                st.error("Please enter a feature name.")
            elif new_feat_name in all_current_features or new_feat_name == "Profit":
                st.error("Invalid or duplicate feature name.")
            else:
                st.session_state.feature_registry.add_binary_feature(new_feat_name, f1, op, f2)
                st.success(f"Feature '{new_feat_name}' created!")
                st.rerun()

    with tab2:
        col_f1_t, col_op1_t, col_f2_t, col_op2_t, col_f3_t = st.columns(5)
        with col_f1_t:
            f1_t = st.selectbox("Feature 1", all_current_features, key="ter_f1")
        with col_op1_t:
            op1_t = st.selectbox("Op 1", ["Addition", "Subtraction", "Multiplication", "Division"], key="ter_op1")
        with col_f2_t:
            f2_t = st.selectbox("Feature 2", all_current_features, key="ter_f2")
        with col_op2_t:
            op2_t = st.selectbox("Op 2", ["Addition", "Subtraction", "Multiplication", "Division"], key="ter_op2")
        with col_f3_t:
            f3_t = st.selectbox("Feature 3", all_current_features, key="ter_f3")
            
        new_feat_name_t = st.text_input("Feature Name", key="ter_name")
            
        if st.button("CREATE FEATURE", key="ter_btn"):
            if not new_feat_name_t:
                st.error("Please enter a feature name.")
            elif new_feat_name_t in all_current_features or new_feat_name_t == "Profit":
                st.error("Invalid or duplicate feature name.")
            else:
                st.session_state.feature_registry.add_ternary_feature(new_feat_name_t, f1_t, op1_t, f2_t, op2_t, f3_t)
                st.success(f"Feature '{new_feat_name_t}' created!")
                st.rerun()

    with tab3:
        f1_un = st.selectbox("Feature", all_current_features, key="un_f1")
        op_un = st.selectbox("Transformation", ["Square", "Square Root", "Log", "Absolute Value"], key="un_op")
        new_feat_name_un = st.text_input("Feature Name", key="un_name")
        
        if st.button("CREATE FEATURE", key="un_btn"):
            if not new_feat_name_un:
                st.error("Please enter a feature name.")
            elif new_feat_name_un in all_current_features or new_feat_name_un == "Profit":
                st.error("Invalid or duplicate feature name.")
            else:
                st.session_state.feature_registry.add_unary_feature(new_feat_name_un, f1_un, op_un)
                st.success(f"Feature '{new_feat_name_un}' created!")
                st.rerun()

st.divider()

# --- Section: MODEL EXPERIMENT ---
st.header("🤖 MODEL EXPERIMENT")

st.subheader("Select Features")
selected_features = []

all_features = st.session_state.feature_registry.get_all_features()

# Make a container for checkboxes to make it look clean
cols_checkbox = st.columns(4)
for idx, feat in enumerate(all_features):
    col = cols_checkbox[idx % 4]
    # Default selection: previous selection if any, else original features
    if st.session_state.latest_selected_features:
        is_default = feat in st.session_state.latest_selected_features
    else:
        is_default = feat in original_features
        
    if col.checkbox(feat, value=is_default, key=f"sel_{feat}"):
        selected_features.append(feat)

st.markdown("**Target**: Profit")

if st.button("🚀 TRAIN MODEL", type="primary"):
    if len(selected_features) == 0:
        st.error("Please select at least one feature.")
    else:
        with st.spinner("Training model..."):
            exp_name = f"Experiment {st.session_state.exp_count}"
            metrics, y_val, y_pred = run_experiment(exp_name, selected_features)
            st.session_state.experiment_tracker.add_experiment(exp_name, selected_features, HARDCODED_MODEL, metrics)
            st.session_state.exp_count += 1
            
            st.session_state.latest_metrics = metrics
            st.session_state.latest_selected_features = selected_features
            st.session_state.latest_y_val = y_val
            st.session_state.latest_y_pred = y_pred

st.divider()

# --- Section: RESULTS ---
if st.session_state.latest_metrics:
    st.header("📈 RESULTS")
    
    st.subheader("Current Performance")
    m_col1, m_col2, m_col3 = st.columns(3)
    
    baseline = st.session_state.baseline_metrics
    latest = st.session_state.latest_metrics
    
    # R2
    r2_pct = ((latest["R²"] - baseline["R²"]) / abs(baseline["R²"])) * 100 if baseline["R²"] != 0 else 0
    m_col1.metric("R² (Higher is better)", f"{latest['R²']:.4f}", f"{r2_pct:+.1f}% vs Baseline")
    
    # MAE
    mae_pct = ((latest["MAE"] - baseline["MAE"]) / baseline["MAE"]) * 100 if baseline["MAE"] != 0 else 0
    m_col2.metric("MAE (Lower is better)", f"{latest['MAE']:.2f}", f"{mae_pct:+.1f}% vs Baseline", delta_color="inverse")
    
    # RMSE
    rmse_pct = ((latest["RMSE"] - baseline["RMSE"]) / baseline["RMSE"]) * 100 if baseline["RMSE"] != 0 else 0
    m_col3.metric("RMSE (Lower is better)", f"{latest['RMSE']:.2f}", f"{rmse_pct:+.1f}% vs Baseline", delta_color="inverse")

st.divider()

# --- Section: EXPERIMENT HISTORY ---
st.header("📊 EXPERIMENT HISTORY")

history_df = st.session_state.experiment_tracker.get_history_df()
# Filter out model column if we don't want to show it
if not history_df.empty and 'Model' in history_df.columns:
    display_df = history_df.drop(columns=['Model'])
else:
    display_df = history_df

st.dataframe(display_df, use_container_width=True)

fig_hist = st.session_state.experiment_tracker.plot_metric_comparison()
if fig_hist:
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# --- Section: EXPERIMENT REPORT ---
st.header("📄 EXPERIMENT REPORT")

# PDF export logic
if st.session_state.latest_metrics:
    try:
        pdf_bytes = generate_pdf_report(
            st.session_state.group_information,
            st.session_state.baseline_metrics,
            display_df,
            st.session_state.latest_metrics,
            st.session_state.latest_selected_features,
            st.session_state.feature_registry
        )
        
        filename = "Feature_Engineering_Report.pdf"
        
        st.download_button(
            label="📄 SAVE EXPERIMENT AS PDF",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
else:
    st.info("Run at least one experiment to generate a PDF report.")

# Reset Experiment
st.markdown("---")
if st.button("Reset Experiment"):
    st.session_state.feature_registry = FeatureRegistry()
    st.session_state.experiment_tracker = ExperimentTracker()
    st.session_state.exp_count = 1
    st.session_state.latest_metrics = None
    st.session_state.latest_selected_features = []
    
    # Re-calculate baseline for the clean state
    bl_metrics, _, _ = run_experiment("Baseline", original_features)
    st.session_state.baseline_metrics = bl_metrics
    st.session_state.experiment_tracker.add_experiment("Baseline", original_features, HARDCODED_MODEL, bl_metrics)
    st.rerun()

