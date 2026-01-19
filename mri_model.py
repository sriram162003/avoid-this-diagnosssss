"""
MRI Stroke Detection Module

This module provides image-based stroke risk analysis for brain MRI scans.

Current implementation uses heuristic feature extraction (intensity patterns,
contrast, edge density) for demonstration purposes. The CNN architecture is
scaffolded for future training with properly annotated medical imaging datasets.

Note: This is an educational tool. Clinical stroke detection requires validated
models trained on expert-annotated datasets with regulatory approval.
"""

import os
import numpy as np
from PIL import Image
import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array

IMG_SIZE = (224, 224)
CLASS_NAMES = ['Normal', 'Stroke']


def create_stroke_cnn_model():
    """Build a VGG16-based CNN for binary stroke classification (scaffolded for future training)."""
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_mri_image(image_data):
    """Load and normalize an MRI image for model input."""
    if isinstance(image_data, bytes):
        image = Image.open(io.BytesIO(image_data))
    else:
        image = Image.open(image_data)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(IMG_SIZE)
    
    img_array = img_to_array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, image

def analyze_mri_features(image):
    """
    Extract statistical features from an MRI scan.
    
    Returns intensity distribution metrics, edge density, and contrast
    measurements that may correlate with abnormal tissue patterns.
    """
    img_array = np.array(image.convert('L'))
    
    mean_intensity = np.mean(img_array)
    std_intensity = np.std(img_array)
    
    high_intensity_ratio = np.sum(img_array > 200) / img_array.size
    low_intensity_ratio = np.sum(img_array < 50) / img_array.size
    
    gradient_x = np.abs(np.diff(img_array, axis=1))
    gradient_y = np.abs(np.diff(img_array, axis=0))
    edge_density = (np.mean(gradient_x) + np.mean(gradient_y)) / 2
    
    features = {
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'high_intensity_ratio': high_intensity_ratio * 100,
        'low_intensity_ratio': low_intensity_ratio * 100,
        'edge_density': edge_density,
        'contrast': std_intensity / (mean_intensity + 1e-6)
    }
    
    return features

def get_simple_prediction(image):
    """
    Compute a heuristic stroke risk score from image features.
    
    This is a demonstration method using thresholds on intensity and texture
    metrics. It does not replace trained medical AI models.
    """
    features = analyze_mri_features(image)
    
    risk_score = 0.0
    
    # High intensity regions may indicate abnormal tissue
    if features['high_intensity_ratio'] > 5:
        risk_score += 0.25
    
    # Elevated contrast can suggest lesions or edema
    if features['contrast'] > 0.5:
        risk_score += 0.2
    
    # Sharp edges may indicate tissue boundaries or damage
    if features['edge_density'] > 15:
        risk_score += 0.2
    
    # High variance often correlates with heterogeneous tissue
    if features['std_intensity'] > 50:
        risk_score += 0.15
    
    if features['mean_intensity'] > 150:
        risk_score += 0.1
    
    return max(0.0, min(1.0, risk_score)), features

class MRIStrokeClassifier:
    """
    Wrapper for MRI-based stroke risk assessment.
    
    Currently uses heuristic feature analysis. The CNN model is available
    for training if annotated MRI datasets become available.
    """
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        
    def load_or_create_model(self):
        """Lazily initialize the CNN model."""
        if self.model is None:
            self.model = create_stroke_cnn_model()
            self.model_loaded = True
        return self.model
    
    def predict(self, image_data):
        """Analyze an MRI image and return stroke risk assessment."""
        img_array, original_image = preprocess_mri_image(image_data)
        
        simple_prob, features = get_simple_prediction(original_image)
        
        prediction = 1 if simple_prob > 0.5 else 0
        
        return {
            'prediction': prediction,
            'class_name': CLASS_NAMES[prediction],
            'stroke_probability': simple_prob,
            'confidence': abs(simple_prob - 0.5) * 2,
            'features': features,
            'processed_image': original_image
        }
    
    def train_on_dataset(self, train_dir, epochs=10, batch_size=32):
        model = self.load_or_create_model()
        
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=batch_size,
            class_mode='binary',
            subset='training'
        )
        
        val_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=batch_size,
            class_mode='binary',
            subset='validation'
        )
        
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator
        )
        
        return history
