# Brain Stroke Prediction System

A machine learning application that predicts stroke risk using patient clinical data and brain MRI scans.

## What it does

This tool helps assess stroke risk through two approaches:

1. **Clinical Data Analysis** — Enter patient information (age, blood pressure, glucose levels, BMI, smoking status) and get a risk prediction based on patterns learned from 5,000+ real patient records.

2. **MRI Scan Analysis** — Upload a brain MRI image to analyze intensity patterns, contrast, and tissue boundaries that may indicate stroke risk.

## Try it out

The app walks you through the full workflow:
- Load the dataset (or bring your own)
- Explore the data with interactive visualizations
- Train a prediction model with adjustable parameters
- Review model performance metrics
- Make predictions on new patients
- Analyze brain scans

## Built with

- **Streamlit** for the web interface
- **scikit-learn** for the machine learning pipeline
- **Plotly** for interactive charts
- **OpenCV & Pillow** for image processing

## Dataset

Uses the [Healthcare Stroke Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle, containing anonymized records from 5,110 patients with 11 clinical features.

## Running locally

```bash
pip install streamlit scikit-learn pandas numpy plotly matplotlib seaborn opencv-python pillow
streamlit run app.py
```

## Important notes

This is an educational project demonstrating ML techniques for medical prediction. It is **not** intended for actual clinical diagnosis. The MRI analysis uses heuristic image features rather than a trained neural network.

Real medical AI systems require:
- Validated training datasets with expert annotations
- Rigorous clinical testing and regulatory approval
- Integration with proper medical imaging equipment

Always consult healthcare professionals for medical advice.

## Project structure

```
├── app.py              # Main application
├── mri_model.py        # MRI image analysis
├── .streamlit/         # Streamlit configuration
└── README.md
```

## License

MIT
