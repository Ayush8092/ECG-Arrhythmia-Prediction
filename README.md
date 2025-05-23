# ECG Arrhythmia Classification using Deep Learning

This project implements a convolutional neural network (CNN) to classify different types of cardiac arrhythmias from ECG signals. The model combines and analyzes data from multiple ECG databases to achieve robust performance.

## Features

- Processes raw ECG data from four major arrhythmia databases
- Implements a customized CNN architecture optimized for ECG classification
- Provides comprehensive model evaluation metrics and visualizations
- Designed for clinical relevance with attention to misclassification risks

## Dataset Sources

The model uses data from four standard ECG databases:
1. MIT-BIH Arrhythmia Database
2. INCART 2-lead Arrhythmia Database
3. MIT-BIH Supraventricular Arrhythmia Database
4. Sudden Cardiac Death Holter Database

## Library Requirements

- Python 3.6+
- TensorFlow 2.x / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

## Implementation Details

### Data Preprocessing
- **Missing value handling**: Dropping missing values (`df.dropna()`)
- **Label encoding**: Converting arrhythmia classes to numerical values using `LabelEncoder()`
- **Feature selection**: Keeping 32 most relevant ECG features (identified through EDA)
- **Standardization**: Using `StandardScaler()` (mean=0, std=1)

### Training Configuration
| Parameter          | Value                          |
|--------------------|--------------------------------|
| Optimizer          | Adam                           |
| Loss function      | Categorical cross-entropy      |
| Batch size         | 256                            |
| Epochs             | 6 (demo), 20+ recommended      |
| Validation split   | 20%                            |
| Random state       | 42 (for reproducibility)       |

### Results

Accuracy of the model is: 0.95184

## Algorithm Comparison

| Algorithm                 | Accuracy  |
|---------------------------|-----------|
| Logistic Regression       | 0.9512    |
| FNN (Feedforward Neural Network) | 0.9825    |
| CNN (Convolutional Neural Network) | 0.98319   |
| XGBoost                  | 0.9828    |
| LightGBM                 | 0.97499   |

