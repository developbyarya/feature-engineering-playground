import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(group_info, baseline_metrics, experiment_history, latest_metrics, selected_features, feature_registry):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    title_style.alignment = 1 # Center
    
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    story = []
    
    # Title
    story.append(Paragraph("FEATURE ENGINEERING PLAYGROUND", title_style))
    story.append(Paragraph("Experiment Report", title_style))
    story.append(Spacer(1, 20))
    
    # Group Information
    story.append(Paragraph("Group Information", heading_style))
    story.append(Paragraph("Members:", normal_style))
    for idx, member in enumerate(group_info.get("members", [])):
        story.append(Paragraph(f"{idx + 1}. {member['name']} - {member['nim']}", normal_style))
    story.append(Spacer(1, 20))
    
    # Dataset Information
    story.append(Paragraph("Dataset Information", heading_style))
    story.append(Paragraph("Dataset: Superstore", normal_style))
    story.append(Paragraph("Rows: 9,994", normal_style))
    story.append(Paragraph("Target: Profit", normal_style))
    story.append(Paragraph("Task: Regression", normal_style))
    story.append(Paragraph("Train / Validation: 80% / 20%", normal_style))
    story.append(Spacer(1, 20))
    
    # Baseline
    story.append(Paragraph("BASELINE MODEL PERFORMANCE", heading_style))
    story.append(Paragraph("Features: Sales, Quantity, Discount, Shipping_Days, Order_Year", normal_style))
    if baseline_metrics:
        story.append(Paragraph(f"R²: {baseline_metrics['R²']:.4f}", normal_style))
        story.append(Paragraph(f"MAE: {baseline_metrics['MAE']:.2f}", normal_style))
        story.append(Paragraph(f"RMSE: {baseline_metrics['RMSE']:.2f}", normal_style))
    story.append(Spacer(1, 20))
    
    # Experiment Result
    story.append(Paragraph("EXPERIMENT RESULT", heading_style))
    if latest_metrics:
        story.append(Paragraph("Selected Features:", normal_style))
        for feat in selected_features:
            story.append(Paragraph(f"✓ {feat}", normal_style))
        story.append(Spacer(1, 10))
        
        # Comparison Table
        data = [
            ["Metric", "Baseline", "Experiment", "Change"]
        ]
        
        r2_pct = ((latest_metrics['R²'] - baseline_metrics['R²']) / abs(baseline_metrics['R²'])) * 100 if baseline_metrics['R²'] != 0 else 0
        mae_pct = ((latest_metrics['MAE'] - baseline_metrics['MAE']) / baseline_metrics['MAE']) * 100 if baseline_metrics['MAE'] != 0 else 0
        rmse_pct = ((latest_metrics['RMSE'] - baseline_metrics['RMSE']) / baseline_metrics['RMSE']) * 100 if baseline_metrics['RMSE'] != 0 else 0
        
        data.append(["R²", f"{baseline_metrics['R²']:.4f}", f"{latest_metrics['R²']:.4f}", f"{r2_pct:+.1f}%"])
        data.append(["MAE", f"{baseline_metrics['MAE']:.2f}", f"{latest_metrics['MAE']:.2f}", f"{mae_pct:+.1f}%"])
        data.append(["RMSE", f"{baseline_metrics['RMSE']:.2f}", f"{latest_metrics['RMSE']:.2f}", f"{rmse_pct:+.1f}%"])
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.white),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("No experiments run yet.", normal_style))
    story.append(Spacer(1, 20))
    
    # Engineered Features
    story.append(Paragraph("ENGINEERED FEATURES", heading_style))
    eng_features = feature_registry.get_engineered_features()
    if eng_features:
        for i, feat in enumerate(eng_features):
            info = feature_registry.get_feature_info(feat)
            story.append(Paragraph(f"{i+1}. {feat}", styles['Heading3']))
            story.append(Paragraph(f"Formula: {info['formula']}", normal_style))
            story.append(Spacer(1, 10))
    else:
        story.append(Paragraph("No engineered features were created.", normal_style))
    story.append(Spacer(1, 20))
    
    # Experiment History
    story.append(Paragraph("EXPERIMENT HISTORY", heading_style))
    if not experiment_history.empty:
        for idx, row in experiment_history.iterrows():
            story.append(Paragraph(f"{row['Experiment']}", styles['Heading3']))
            story.append(Paragraph(f"Features: {row['Features']}", normal_style))
            story.append(Paragraph(f"R²: {row['R²']:.4f}", normal_style))
            story.append(Spacer(1, 10))
    else:
        story.append(Paragraph("No experiment history.", normal_style))
    story.append(Spacer(1, 20))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
