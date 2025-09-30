import streamlit as st
import pandas as pd
import numpy as np
from utils.ml_models import MLModels
from utils.visualizations import Visualizations

st.set_page_config(page_title="Churn Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Customer Churn Prediction")
st.markdown("Train machine learning models to predict customer churn and analyze model performance.")

# Initialize components if not exists
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = MLModels()
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = Visualizations()

# Check if data is available
if 'data_processor' not in st.session_state or st.session_state.data_processor.processed_features is None:
    st.error("❌ No processed data available. Please upload and preprocess your data first.")
    st.page_link("pages/1_Data_Upload.py", label="📁 Go to Data Upload", icon="📁")
    st.stop()

# Model training section
st.header("🤖 Model Training")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Basic Models")
    st.write("**Logistic Regression**: Fast, interpretable linear model")
    st.write("**Random Forest**: Ensemble method with feature importance")

with col2:
    if st.button("🚀 Train Basic Models", type="primary"):
        with st.spinner("Preparing data and training models..."):
            # Prepare data for churn prediction
            if st.session_state.ml_models.prepare_churn_data(st.session_state.data_processor.processed_features):
                # Train models
                if st.session_state.ml_models.train_churn_models():
                    st.success("✅ Basic models trained successfully!")
                    st.rerun()

# Advanced models section
if st.session_state.ml_models.lr_model is not None:
    st.markdown("---")
    st.header("🚀 Advanced Models")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Gradient Boosting")
        st.write("**XGBoost**: Extreme Gradient Boosting")
        st.write("**LightGBM**: Light Gradient Boosting Machine")
        if st.button("Train Advanced Models"):
            with st.spinner("Training XGBoost and LightGBM..."):
                if st.session_state.ml_models.train_advanced_models():
                    st.rerun()
    
    with col2:
        st.subheader("Deep Learning")
        st.write("**ANN**: Artificial Neural Network")
        st.write("**RNN**: Recurrent Neural Network")
        if st.button("Train Deep Learning Models"):
            with st.spinner("Training deep learning models (this may take a minute)..."):
                if st.session_state.ml_models.train_deep_learning_models():
                    st.rerun()
    
    with col3:
        st.subheader("Model Status")
        if st.session_state.ml_models.xgb_model is not None:
            st.success("✅ XGBoost trained")
        if st.session_state.ml_models.lgb_model is not None:
            st.success("✅ LightGBM trained")
        if st.session_state.ml_models.ann_model is not None:
            st.success("✅ ANN trained")
        if st.session_state.ml_models.rnn_model is not None:
            st.success("✅ RNN trained")

# Model performance section
if st.session_state.ml_models.lr_model is not None and st.session_state.ml_models.rf_model is not None:
    st.markdown("---")
    st.header("📊 Model Performance")
    
    # Get performance metrics
    metrics = st.session_state.ml_models.get_model_performance()
    
    if metrics:
        # Performance metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Logistic Regression")
            lr_metrics = metrics['Logistic Regression']
            
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Accuracy", f"{lr_metrics['accuracy']:.3f}")
                st.metric("Precision", f"{lr_metrics['precision']:.3f}")
            with metric_col2:
                st.metric("Recall", f"{lr_metrics['recall']:.3f}")
                st.metric("F1-Score", f"{lr_metrics['f1_score']:.3f}")
            
            # Confusion matrix
            lr_cm_fig = st.session_state.visualizations.plot_confusion_matrix(
                lr_metrics['confusion_matrix'], 'Logistic Regression'
            )
            st.plotly_chart(lr_cm_fig, use_container_width=True)
        
        with col2:
            st.subheader("🌲 Random Forest")
            rf_metrics = metrics['Random Forest']
            
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Accuracy", f"{rf_metrics['accuracy']:.3f}")
                st.metric("Precision", f"{rf_metrics['precision']:.3f}")
            with metric_col2:
                st.metric("Recall", f"{rf_metrics['recall']:.3f}")
                st.metric("F1-Score", f"{rf_metrics['f1_score']:.3f}")
            
            # Confusion matrix
            rf_cm_fig = st.session_state.visualizations.plot_confusion_matrix(
                rf_metrics['confusion_matrix'], 'Random Forest'
            )
            st.plotly_chart(rf_cm_fig, use_container_width=True)
        
        # Model comparison
        st.subheader("⚖️ Model Comparison")
        comparison_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Logistic Regression': [
                lr_metrics['accuracy'],
                lr_metrics['precision'], 
                lr_metrics['recall'],
                lr_metrics['f1_score']
            ],
            'Random Forest': [
                rf_metrics['accuracy'],
                rf_metrics['precision'],
                rf_metrics['recall'], 
                rf_metrics['f1_score']
            ]
        }
        
        # Add advanced models if available
        if 'XGBoost' in metrics:
            xgb_metrics = metrics['XGBoost']
            comparison_data['XGBoost'] = [
                xgb_metrics['accuracy'],
                xgb_metrics['precision'],
                xgb_metrics['recall'],
                xgb_metrics['f1_score']
            ]
        
        if 'LightGBM' in metrics:
            lgb_metrics = metrics['LightGBM']
            comparison_data['LightGBM'] = [
                lgb_metrics['accuracy'],
                lgb_metrics['precision'],
                lgb_metrics['recall'],
                lgb_metrics['f1_score']
            ]
        
        if 'ANN (Deep Learning)' in metrics:
            ann_metrics = metrics['ANN (Deep Learning)']
            comparison_data['ANN'] = [
                ann_metrics['accuracy'],
                ann_metrics['precision'],
                ann_metrics['recall'],
                ann_metrics['f1_score']
            ]
        
        if 'RNN (Deep Learning)' in metrics:
            rnn_metrics = metrics['RNN (Deep Learning)']
            comparison_data['RNN'] = [
                rnn_metrics['accuracy'],
                rnn_metrics['precision'],
                rnn_metrics['recall'],
                rnn_metrics['f1_score']
            ]
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Highlight best model
        st.subheader("🏆 Best Model")
        best_model = comparison_df.iloc[0, 1:].idxmax()  # Best by accuracy
        best_accuracy = comparison_df.iloc[0, 1:].max()
        st.success(f"**{best_model}** achieves the highest accuracy of {best_accuracy:.3f}")
        
        # Feature importance
        feature_importance = st.session_state.ml_models.get_feature_importance()
        if feature_importance is not None:
            st.markdown("---")
            st.subheader("🎯 Feature Importance")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                importance_fig = st.session_state.visualizations.plot_feature_importance(feature_importance)
                st.plotly_chart(importance_fig, use_container_width=True)
            
            with col2:
                st.write("**Top 10 Most Important Features:**")
                top_features = feature_importance.head(10)
                for idx, row in top_features.iterrows():
                    st.write(f"**{row['feature']}**: {row['importance']:.3f}")

# Churn predictions section
if st.session_state.ml_models.rf_model is not None:
    st.markdown("---")
    st.header("🎯 Churn Predictions")
    
    # Get predictions
    predictions = st.session_state.ml_models.predict_churn_probabilities()
    
    if predictions is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Churn Probability Distribution")
            prob_dist_fig = st.session_state.visualizations.plot_churn_probability_distribution(
                predictions['churn_probability']
            )
            st.plotly_chart(prob_dist_fig, use_container_width=True)
        
        with col2:
            st.subheader("🚨 Risk Segments")
            risk_segments_fig = st.session_state.visualizations.plot_churn_risk_segments(predictions)
            st.plotly_chart(risk_segments_fig, use_container_width=True)
        
        # High-risk customers
        st.subheader("⚠️ High-Risk Customers")
        
        high_risk_threshold = st.slider(
            "Churn Probability Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Customers above this threshold are considered high-risk"
        )
        
        high_risk_customers = predictions[predictions['churn_probability'] >= high_risk_threshold]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("High-Risk Customers", len(high_risk_customers))
        with col2:
            total_customers = len(predictions)
            risk_percentage = (len(high_risk_customers) / total_customers * 100) if total_customers > 0 else 0
            st.metric("Risk Percentage", f"{risk_percentage:.1f}%")
        with col3:
            avg_risk_score = high_risk_customers['churn_probability'].mean() if len(high_risk_customers) > 0 else 0
            st.metric("Avg Risk Score", f"{avg_risk_score:.3f}")
        
        if len(high_risk_customers) > 0:
            st.write("**High-Risk Customer Analysis:**")
            
            # Show high-risk customers table
            display_predictions = predictions.copy()
            display_predictions['risk_level'] = pd.cut(
                display_predictions['churn_probability'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low', 'Medium', 'High']
            )
            
            high_risk_display = display_predictions[
                display_predictions['churn_probability'] >= high_risk_threshold
            ].sort_values('churn_probability', ascending=False)
            
            st.dataframe(
                high_risk_display[['churn_probability', 'churn_prediction', 'risk_level']].round(3),
                use_container_width=True
            )
            
            # Retention recommendations
            st.subheader("💡 Retention Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Immediate Actions:**")
                st.write("• Personal outreach to high-risk customers")
                st.write("• Offer loyalty programs or discounts")
                st.write("• Conduct satisfaction surveys")
                st.write("• Provide premium customer support")
            
            with col2:
                st.write("**Long-term Strategies:**")
                st.write("• Improve product features based on feedback")
                st.write("• Develop targeted marketing campaigns")
                st.write("• Create customer success programs")
                st.write("• Monitor engagement metrics regularly")

# Download predictions
if st.session_state.ml_models.rf_model is not None and predictions is not None:
    st.markdown("---")
    st.subheader("💾 Export Predictions")
    
    # Prepare download data
    download_data = predictions.copy()
    download_data['customer_index'] = download_data.index
    download_data = download_data[['customer_index', 'churn_probability', 'churn_prediction']]
    
    csv_data = download_data.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Churn Predictions (CSV)",
        data=csv_data,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.page_link("pages/1_Data_Upload.py", label="📁 Data Upload", icon="📁")
with col2:
    st.page_link("pages/3_Customer_Segmentation.py", label="👥 Customer Segmentation", icon="👥")
with col3:
    st.page_link("pages/4_Sales_Analysis.py", label="📈 Sales Analysis", icon="📈")
