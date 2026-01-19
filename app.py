"""
Brain Stroke Prediction Application

A clinical decision support tool that combines patient health record analysis
with brain MRI imaging to assess stroke risk. Uses a Random Forest classifier
trained on the Kaggle Healthcare Stroke Dataset.

Author: [Your Name]
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
import io
from PIL import Image
from mri_model import MRIStrokeClassifier, analyze_mri_features

st.set_page_config(
    page_title="Brain Stroke Prediction",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5a6c7d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .info-box {
        background-color: #f0f7ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Brain Stroke Risk Assessment</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">A clinical decision support tool combining patient data analysis with medical imaging</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **What this tool does:**
    - Analyzes patient health records to identify stroke risk factors
    - Processes brain MRI scans for visual indicators
    - Provides probability-based risk assessment
    """)
with col2:
    st.markdown("""
    **Built for:**
    - Medical research and education
    - Clinical workflow demonstration
    - Healthcare data analysis practice
    """)

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

DATASET_URL = "https://gist.githubusercontent.com/aishwarya8615/d2107f828d3f904839cbcb7eaa85bd04/raw/cec0340503d82d270821e03254993b6dede60afb/healthcare-dataset-stroke-data.csv"

@st.cache_data
def load_kaggle_dataset():
    """Fetch the Healthcare Stroke Dataset from Kaggle (cached)."""
    try:
        df = pd.read_csv(DATASET_URL)
        df = df.drop('id', axis=1, errors='ignore')
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        return df, True
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Feature Engineering & Preprocessing
# ---------------------------------------------------------------------------

def get_feature_columns(df):
    """Separate numeric and categorical columns for preprocessing."""
    target_col = 'stroke'
    feature_cols = [col for col in df.columns if col != target_col]
    
    numeric_cols = df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    return feature_cols, numeric_cols, categorical_cols


def create_preprocessor(numeric_cols, categorical_cols):
    """
    Build a sklearn ColumnTransformer for mixed data types.
    
    Numeric features: median imputation + standardization
    Categorical features: mode imputation + one-hot encoding
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    return preprocessor


def validate_uploaded_data(df):
    """Check that an uploaded CSV has the required columns."""
    required_columns = [
        'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
        'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if 'stroke' not in df.columns:
        return False, "Dataset must contain a 'stroke' column as the target variable."
    
    if missing_cols:
        return False, f"Dataset is missing required columns: {', '.join(missing_cols)}"
    
    return True, "Dataset validation successful."


# ---------------------------------------------------------------------------
# Model Training & Evaluation
# ---------------------------------------------------------------------------

def train_model(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    """Train a Random Forest classifier with balanced class weights."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Compute classification metrics, confusion matrix, and ROC curve data."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0)
    }
    
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    return metrics, cm, fpr, tpr, roc_auc, y_pred


# ---------------------------------------------------------------------------
# Application Interface
# ---------------------------------------------------------------------------

st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Explorer", "Model Training", "Performance", "Risk Calculator", "MRI Analysis"
])

with tab1:
    st.subheader("Dataset")
    st.write("Choose a data source to begin. The Kaggle dataset contains real anonymized patient records from healthcare providers.")
    
    data_source = st.radio(
        "Data source",
        ["Kaggle Healthcare Dataset (5,110 patients)", "Upload my own CSV"]
    )
    
    if data_source == "Upload my own CSV":
        uploaded_file = st.file_uploader("Drop your CSV file here", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df = df.drop('id', axis=1, errors='ignore')
            st.success("File loaded successfully")
        else:
            st.info("Your CSV should include a 'stroke' column (0 or 1) as the target variable.")
            df = None
    else:
        df, status = load_kaggle_dataset()
        if df is not None:
            st.success("Dataset ready - 5,110 patient records loaded")
        else:
            st.error(f"Could not load dataset: {status}")
    
    if df is not None:
        st.session_state['df'] = df
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            if 'stroke' in df.columns:
                stroke_rate = df['stroke'].mean() * 100
                st.metric("Stroke Rate", f"{stroke_rate:.2f}%")
        with col3:
            st.metric("Features", len(df.columns) - 1)
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        st.subheader("Data Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.write("**Explore the Data**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'stroke' in df.columns:
                fig = px.pie(df, names='stroke', title='Cases with vs without stroke',
                           color_discrete_sequence=['#2ecc71', '#e74c3c'])
                fig.update_traces(labels=['No Stroke', 'Stroke'])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'age' in df.columns:
                fig = px.histogram(df, x='age', color='stroke' if 'stroke' in df.columns else None,
                                 title='Patient ages in the dataset',
                                 color_discrete_sequence=['#3498db', '#e74c3c'])
                st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            if 'avg_glucose_level' in df.columns:
                fig = px.box(df, x='stroke' if 'stroke' in df.columns else None, 
                           y='avg_glucose_level',
                           title='Glucose levels: stroke vs no stroke',
                           color='stroke' if 'stroke' in df.columns else None,
                           color_discrete_sequence=['#2ecc71', '#e74c3c'])
                st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            if 'bmi' in df.columns:
                fig = px.box(df, x='stroke' if 'stroke' in df.columns else None, 
                           y='bmi',
                           title='BMI by Stroke Status',
                           color='stroke' if 'stroke' in df.columns else None,
                           color_discrete_sequence=['#2ecc71', '#e74c3c'])
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Model Training")
    
    if 'df' not in st.session_state or st.session_state['df'] is None:
        st.warning("Please load data first in the 'Data' tab.")
    else:
        df = st.session_state['df']
        
        if 'stroke' not in df.columns:
            st.error("Dataset must contain a 'stroke' column as the target variable.")
        else:
            st.write("Adjust the parameters below to customize the model, or use the defaults for a good starting point.")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_estimators = st.slider("Trees in forest", 10, 500, 100, 10, help="More trees = better accuracy but slower training")
            with col2:
                max_depth = st.slider("Tree depth", 2, 50, 10, 1, help="Deeper trees can capture complex patterns")
            with col3:
                test_size = st.slider("Validation split", 10, 40, 20, 5, help="Percentage of data held back for testing") / 100
            
            if st.button("Train Model", type="primary"):
                with st.spinner("Processing data and training model..."):
                    feature_cols, numeric_cols, categorical_cols = get_feature_columns(df)
                    
                    X = df[feature_cols]
                    y = df['stroke']
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                    
                    preprocessor = create_preprocessor(numeric_cols, categorical_cols)
                    
                    X_train_processed = preprocessor.fit_transform(X_train)
                    X_test_processed = preprocessor.transform(X_test)
                    
                    model = train_model(X_train_processed, y_train, n_estimators, max_depth)
                    
                    metrics, cm, fpr, tpr, roc_auc, y_pred = evaluate_model(model, X_test_processed, y_test)
                    
                    st.session_state['model'] = model
                    st.session_state['preprocessor'] = preprocessor
                    st.session_state['feature_cols'] = feature_cols
                    st.session_state['numeric_cols'] = numeric_cols
                    st.session_state['categorical_cols'] = categorical_cols
                    st.session_state['metrics'] = metrics
                    st.session_state['confusion_matrix'] = cm
                    st.session_state['roc_data'] = (fpr, tpr, roc_auc)
                    st.session_state['y_test'] = y_test
                    st.session_state['y_pred'] = y_pred
                    
                    st.success("Done! Model is ready to use.")
                    
                    st.write("**Results**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                    col2.metric("Precision", f"{metrics['Precision']:.4f}")
                    col3.metric("Recall", f"{metrics['Recall']:.4f}")
                    col4.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
                    
                    st.subheader("Feature Importance")
                    feature_names_transformed = preprocessor.get_feature_names_out()
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names_transformed,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=True).tail(15)
                    
                    fig = px.bar(feature_importance, x='Importance', y='Feature',
                               orientation='h', title='Which factors matter most',
                               color='Importance',
                               color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Model Performance")
    
    if 'model' not in st.session_state:
        st.info("Train a model first to see evaluation metrics here.")
    else:
        metrics = st.session_state['metrics']
        cm = st.session_state['confusion_matrix']
        fpr, tpr, roc_auc = st.session_state['roc_data']
        
        st.write("**Key Metrics**")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        col2.metric("Precision", f"{metrics['Precision']:.4f}")
        col3.metric("Recall", f"{metrics['Recall']:.4f}")
        col4.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Prediction Accuracy Breakdown**")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['No Stroke', 'Stroke'],
                       yticklabels=['No Stroke', 'Stroke'])
            ax.set_xlabel('What the model predicted')
            ax.set_ylabel('What actually happened')
            ax.set_title('')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("ROC Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                    name=f'ROC Curve (AUC = {roc_auc:.4f})',
                                    line=dict(color='#3498db', width=2)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                    name='Random Classifier',
                                    line=dict(color='gray', width=1, dash='dash')))
            fig.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Model Interpretation")
        st.markdown(f"""
        **Model Performance Summary:**
        
        - **Accuracy ({metrics['Accuracy']:.2%})**: The model correctly classifies {metrics['Accuracy']:.2%} of all cases.
        - **Precision ({metrics['Precision']:.2%})**: When the model predicts a stroke, it is correct {metrics['Precision']:.2%} of the time.
        - **Recall ({metrics['Recall']:.2%})**: The model identifies {metrics['Recall']:.2%} of actual stroke cases.
        - **F1-Score ({metrics['F1-Score']:.2%})**: The harmonic mean of precision and recall.
        - **AUC ({roc_auc:.4f})**: The model's ability to distinguish between stroke and non-stroke cases.
        """)

with tab4:
    st.subheader("Individual Risk Assessment")
    
    if 'model' not in st.session_state:
        st.info("Train a model first using the Model Training tab, then come back here to make predictions.")
    else:
        st.write("Enter the patient's clinical information below to calculate their stroke risk.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Demographics**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
            ever_married = st.selectbox("Marital status", ["Yes", "No"], format_func=lambda x: "Married" if x == "Yes" else "Single")
        
        with col2:
            st.write("**Medical History**")
            hypertension = st.selectbox("High blood pressure", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            heart_disease = st.selectbox("Heart disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            smoking_status = st.selectbox("Smoking", ["never smoked", "formerly smoked", "smokes", "Unknown"])
        
        with col3:
            st.write("**Clinical Measurements**")
            avg_glucose_level = st.number_input("Glucose level (mg/dL)", min_value=50.0, max_value=300.0, value=100.0)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
            work_type = st.selectbox("Occupation", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
            residence_type = st.selectbox("Location", ["Urban", "Rural"])
        
        if st.button("Calculate Risk", type="primary"):
            input_data = pd.DataFrame({
                'gender': [gender],
                'age': [age],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'ever_married': [ever_married],
                'work_type': [work_type],
                'Residence_type': [residence_type],
                'avg_glucose_level': [avg_glucose_level],
                'bmi': [bmi],
                'smoking_status': [smoking_status]
            })
            
            feature_cols = st.session_state['feature_cols']
            input_data = input_data.reindex(columns=feature_cols)
            
            preprocessor = st.session_state['preprocessor']
            input_processed = preprocessor.transform(input_data)
            
            model = st.session_state['model']
            prediction = model.predict(input_processed)[0]
            probability = model.predict_proba(input_processed)[0]
            
            st.subheader("Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ HIGH STROKE RISK DETECTED")
                    st.markdown(f"**Stroke Probability: {probability[1]:.2%}**")
                else:
                    st.success("✅ LOW STROKE RISK")
                    st.markdown(f"**Stroke Probability: {probability[1]:.2%}**")
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability[1] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Stroke Risk (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred" if probability[1] > 0.5 else "green"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Risk Factors Analysis")
            risk_factors = []
            if age > 60:
                risk_factors.append("Advanced age (>60 years)")
            if hypertension == 1:
                risk_factors.append("Hypertension")
            if heart_disease == 1:
                risk_factors.append("Heart disease")
            if avg_glucose_level > 150:
                risk_factors.append("High glucose level (>150 mg/dL)")
            if bmi > 30:
                risk_factors.append("Obesity (BMI > 30)")
            if smoking_status == "smokes":
                risk_factors.append("Current smoker")
            
            if risk_factors:
                st.warning("**Identified Risk Factors:**")
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
            else:
                st.info("No significant risk factors identified based on the provided information.")

with tab5:
    st.subheader("Brain Scan Analysis")
    st.write("Upload a brain MRI image to analyze it for potential stroke indicators. The system examines intensity patterns, contrast levels, and tissue boundaries.")
    
    st.caption("This is an educational demonstration using image feature analysis. Clinical diagnosis requires validated medical AI systems and professional interpretation.")
    
    if 'mri_classifier' not in st.session_state:
        st.session_state['mri_classifier'] = MRIStrokeClassifier()
    
    uploaded_mri = st.file_uploader(
        "Choose an MRI image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Accepts axial, sagittal, or coronal views"
    )
    
    if uploaded_mri is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Your scan**")
            image = Image.open(uploaded_mri)
            st.image(image, caption="Uploaded MRI Image", use_container_width=True)
        
        if st.button("🔍 Analyze MRI Scan", type="primary"):
            with st.spinner("Analyzing MRI scan..."):
                uploaded_mri.seek(0)
                classifier = st.session_state['mri_classifier']
                result = classifier.predict(uploaded_mri)
                
                with col2:
                    st.subheader("Analysis Result")
                    
                    if result['prediction'] == 1:
                        st.error(f"⚠️ STROKE INDICATORS DETECTED")
                        st.markdown(f"**Stroke Probability: {result['stroke_probability']:.2%}**")
                    else:
                        st.success(f"✅ NO STROKE INDICATORS DETECTED")
                        st.markdown(f"**Stroke Probability: {result['stroke_probability']:.2%}**")
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result['stroke_probability'] * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Stroke Risk (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkred" if result['stroke_probability'] > 0.5 else "green"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Image Analysis Features")
                features = result['features']
                
                col3, col4, col5 = st.columns(3)
                with col3:
                    st.metric("Mean Intensity", f"{features['mean_intensity']:.1f}")
                    st.metric("Std Deviation", f"{features['std_intensity']:.1f}")
                with col4:
                    st.metric("High Intensity Ratio", f"{features['high_intensity_ratio']:.2f}%")
                    st.metric("Low Intensity Ratio", f"{features['low_intensity_ratio']:.2f}%")
                with col5:
                    st.metric("Edge Density", f"{features['edge_density']:.2f}")
                    st.metric("Contrast", f"{features['contrast']:.3f}")
                
                st.subheader("Interpretation")
                st.markdown("""
                **Analysis Notes:**
                - **High Intensity Regions**: May indicate areas of acute ischemia or hemorrhage
                - **Edge Density**: Higher values may suggest tissue abnormalities
                - **Contrast**: Significant contrast variations may indicate lesions
                
                **Important Disclaimer:** This AI analysis is for educational purposes only. 
                Always consult a qualified radiologist or neurologist for proper diagnosis.
                """)
    else:
        st.info("Please upload a brain MRI scan image to begin analysis.")
        
        st.subheader("Supported Image Formats")
        st.markdown("""
        - PNG, JPG, JPEG, BMP, TIFF
        - Recommended: High-resolution axial brain MRI slices
        - Best results with T1, T2, or FLAIR weighted images
        """)
        
        st.subheader("What the Analysis Measures")
        st.markdown("""
        The image feature analysis examines:
        - **Intensity patterns**: Mean and variation of pixel brightness
        - **High/Low intensity ratios**: Percentage of very bright or dark regions
        - **Edge density**: Amount of tissue boundary definition
        - **Contrast**: Variation in intensity levels across the image
        """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 1rem 0;'>
    <p style='margin-bottom: 0.5rem;'>Developed for medical research and educational purposes</p>
    <p style='font-size: 0.85rem;'>This tool is not intended for clinical diagnosis. Please consult qualified healthcare professionals for medical advice.</p>
</div>
""", unsafe_allow_html=True)
