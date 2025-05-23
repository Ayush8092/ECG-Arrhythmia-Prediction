# ECG Arrhythmia Classification using Deep Learning

This study employs a convolutional neural network (CNN) to classify different cardiac arrhythmias from electrocardiogram (ECG) signals. The model processes and examines data from different ECG databases to deliver high performance.

## Characteristics

- Converts raw ECG signals of four large arrhythmia databases
- Makes use of a specialized CNN architecture designed for classifying ECG
- Includes complete model assessment statistics and plots
- Structured for clinical application with emphasis on risks of misclassification

## Dataset Sources

The model is trained on four normal ECG databases:

1. MIT-BIH Arrhythmia Database
2. INCART 2-lead Arrhythmia Database
3. MIT-BIH Supraventricular Arrhythmia Database
4. Sudden Cardiac Death Holter Database

## Library Prerequisites

- Python 3.6+
- TensorFlow 2.x / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

## Implementation Details

### Data Loading and Preprocessing

The project starts by importing four different sets of ECG data in CSV format. Each data set has a unique set of ECG signals, and data sets are merged to form a large, heterogeneous data set. This is done to improve model generalization since the merged data represents variability in ECG patterns within and between different patient groups and populations.

Preprocessing includes:

- **Handling Missing Data**: Records with missing values are removed to maintain the dataset's integrity.
- **Encoding Labels**: Text labels of arrhythmia types are converted to numerical labels through a label encoding. Numerical representation is needed for feeding data to the CNN model.
- **Feature Selection**: Only the most representative electrocardiogram (ECG) features, determined by exploratory data analysis (EDA), are retained. Features irrelevant to arrhythmia classification are discarded.
- **Standardization**: The features retained are standardized to a mean of 0 and a standard deviation of 1. Standardization maintains all features at the same scale, which is important for CNN model convergence.

### Data Partitioning

The data set is divided into a training and a test set based on an 80/20 ratio. Splitting is done randomly based on a fixed random state to facilitate reproducibility. Model learning is done based on training data, and model performance is measured based on the test data on new samples.

### Model Architecture

The CNN architecture is such that it can effectively process ECG signals:

- **Input Layer**: The input data is converted into a 2D form (8×4) to maintain both morphological and temporal patterns of the ECG signals.
- **Convolutional Layers**: These layers learn from ECG signal features, beginning with low-level waveform identification and moving up to higher-level patterns.
- **Batch normalization** is used to improve the stability of the training process, while the ELU activation function gives smooth gradient flow and performs better in learning compared to the ReLU activation function.
- **Pooling Layers**: Max pooling reduces spatial dimensions, which helps reduce computational complexity and prevent overfitting.
- **Fully Connected Layers**: These layers add the extracted features and pass them to the output layer to be classified.
- **Output Layer**: A softmax activation function gives a probability distribution between the given arrhythmia classes.

### Model Compilation and Training

The model uses categorical cross-entropy as the loss function, Adam optimizer, and accuracy as the metric for overall evaluation. The model is trained with the minimum number of epochs for demonstration purposes. The training data is set aside for validation to track the performance of the model during training.

### Model Evaluation

The model’s performance is evaluated using:

1. **Quantitative Metrics**: Test loss and accuracy measure how reliable the model is overall in arrhythmia classification.
2. **Confusion Matrix**: It gives detailed information regarding the model's capacity to distinguish between various classes of arrhythmia.
3. **Classification Report**: Involves computing metrics such as precision, recall, and F1-score for each individual class.

### Visualization

- **Training Progress**: Graphical plots of loss and accuracy throughout epochs depict trends in model development and possible overfitting.
- **Confusion Matrix**: A heatmap graphically identifies the model's strengths and weaknesses in arrhythmia classification.


### Results

Accuracy of the model is: 0.98318

## Algorithm Comparison

| Algorithm                 | Accuracy  |
|---------------------------|-----------|
| Logistic Regression       | 0.9512    |
| FNN (Feedforward Neural Network) | 0.9825    |
| CNN (Convolutional Neural Network) | 0.98319   |
| XGBoost                  | 0.9828    |
| LightGBM                 | 0.97499   |

