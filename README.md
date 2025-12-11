# Heart Disease Risk Prediction Using Machine Learning: An Explainable AI Approach

## Abstract

This repository presents a comprehensive machine learning framework for predicting heart disease risk using the UCI Heart Disease dataset. The project employs a Random Forest classifier and integrates advanced explainability techniques, including SHAP (SHapley Additive exPlanations) values, Partial Dependence Plots (PDP), and permutation importance analysis, to provide interpretable and actionable insights into cardiovascular disease prediction.

## Introduction

Cardiovascular diseases remain a leading cause of mortality worldwide. Early and accurate prediction of heart disease risk can significantly improve patient outcomes through timely intervention. This project addresses the challenge of developing both accurate and interpretable machine learning models for clinical decision support. By leveraging ensemble learning methods and state-of-the-art explainability tools, we aim to bridge the gap between predictive performance and model transparency in healthcare applications.

## Dataset

The **UCI Heart Disease Dataset** is a well-established benchmark dataset in medical machine learning research. The dataset contains 1,025 patient records with 14 clinical features:

### Features

1. **age**: Patient age in years
2. **sex**: Gender (0 = female, 1 = male)
3. **cp**: Chest pain type
   - 0: Typical angina
   - 1: Atypical angina
   - 2: Non-anginal pain
   - 3: Asymptomatic
4. **trestbps**: Resting blood pressure (mm Hg)
5. **chol**: Serum cholesterol (mg/dl)
6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. **restecg**: Resting electrocardiographic results
   - 0: Normal
   - 1: ST-T wave abnormality
   - 2: Left ventricular hypertrophy
8. **thalach**: Maximum heart rate achieved
9. **exang**: Exercise induced angina (1 = yes, 0 = no)
10. **oldpeak**: ST depression induced by exercise relative to rest
11. **slope**: Slope of the peak exercise ST segment
   - 0: Upsloping
   - 1: Flat
   - 2: Downsloping
12. **ca**: Number of major vessels (0-3) colored by fluoroscopy
13. **thal**: Thalassemia
   - 0: Unknown
   - 1: Normal
   - 2: Fixed defect
   - 3: Reversible defect
14. **target**: Presence of heart disease (1 = disease, 0 = no disease)

### Dataset Characteristics

- **Total samples**: 1,025
- **Features**: 14 (13 input features + 1 target variable)
- **Missing values**: None
- **Class distribution**: Balanced dataset

## Methodology

### 1. Data Preprocessing

- **Categorical encoding**: Converted integer-encoded categorical features to descriptive string labels
- **One-hot encoding**: Transformed categorical variables into binary features for machine learning compatibility
- **Data splitting**: 80% training set (820 samples) and 20% test set (205 samples) with random state = 10

### 2. Exploratory Data Analysis (EDA)

- Comprehensive statistical analysis using `ydata-profiling`
- Correlation analysis and heatmap visualization
- Distribution analysis of individual features
- Feature-target relationship visualization using:
  - Bar charts
  - Violin plots
  - Box plots
  - Scatter plots
  - Pair plots

### 3. Model Development

**Random Forest Classifier** was selected as the primary model due to its:
- Robustness to overfitting
- Ability to handle non-linear relationships
- Built-in feature importance estimation
- Compatibility with tree-based explainability methods

**Hyperparameters**:
- `max_depth`: 5
- `n_estimators`: 100
- `random_state`: 5

### 4. Model Evaluation

Performance metrics calculated on the test set:
- **Confusion Matrix**: Visual representation of classification performance
- **Classification Report**: Precision, Recall, F1-score for each class
- **ROC Curve**: Receiver Operating Characteristic curve analysis
- **AUC-ROC Score**: Area Under the ROC Curve

### 5. Model Explainability

#### 5.1 SHAP (SHapley Additive exPlanations) Analysis

- **TreeExplainer**: Calculated SHAP values for all test set samples
- **Summary plots**: Global feature importance visualization
- **Force plots**: Individual prediction explanations
- **Dependence plots**: Feature interaction analysis
- **Decision plots**: Decision path visualization for predictions

#### 5.2 Partial Dependence Plots (PDP)

- Analyzed marginal effect of individual features on predictions
- Identified feature interaction effects using `pdpbox`

#### 5.3 Permutation Importance

- Evaluated feature importance through permutation-based analysis using `eli5`

## Results

### Model Performance

The Random Forest classifier achieved the following performance metrics on the test set:

- **Accuracy**: 90%
- **Precision**: 87% (Healthy), 89% (Disease)
- **Recall**: 87% (Healthy), 94% (Disease)
- **F1-Score**: 89% (Healthy), 91% (Disease)
- **ROC-AUC Score**: 0.89

### Key Findings

1. **Top Contributing Features** (based on SHAP analysis):
   - `chest_pain_type_typical angina`: Highest importance (0.149)
   - `thalassemia_fixed defect`: 0.112
   - `ST_depression`: 0.109
   - `major_vessels_num`: 0.107
   - `maximum_heart_rate`: 0.085

2. **Feature Relationships**:
   - Higher maximum heart rate positively correlates with disease prediction
   - Greater number of major vessels reduces disease likelihood
   - Chest pain type (typical angina) is the strongest predictor

3. **Model Interpretability**:
   - SHAP values successfully identified critical risk factors
   - Individual predictions can be explained through force plots
   - Feature interactions revealed through dependence plots

## Project Structure

```
Heart-Disease-Dataset/
│
├── 0_package.ipynb                          # Package installation and environment setup
├── 0_datasetInfo.ipynb                      # Dataset information and initial exploration
├── 1_dataSimpleAnalysisVisualization.ipynb  # Exploratory data analysis and visualization
├── 2_dataPreprocessing.ipynb                # Data preprocessing and feature engineering
├── 3_buildRandomForestClassificationModel.ipynb  # Random Forest model construction
├── 4_testSetDataPredictionAndModelPerformanceEvaluation.ipynb  # Model evaluation
├── 5_ TrainingModelAndTestSamplePrediction.ipynb  # Training and prediction pipeline
├── 6_ShapValue.ipynb                        # SHAP value calculation
├── 7_VisualAnalysisOfShapValues.ipynb       # SHAP visualization and analysis
├── PermutationImportance.ipynb              # Permutation importance analysis
├── pdpbox_0.ipynb                           # Partial Dependence Plot analysis (Part 1)
├── pdpbox_1.ipynb                           # Partial Dependence Plot analysis (Part 2)
│
├── heart.csv                                # Original UCI Heart Disease dataset
├── heart_disease_uci.csv                    # Alternative dataset file
├── process_heart.csv                        # Preprocessed dataset (one-hot encoded)
│
├── profile.html                             # Automated EDA report (ydata-profiling)
├── tree.png                                 # Visualized decision tree
├── tree.dot                                 # Decision tree graph representation
├── heartDiseaseAndAges.png                  # Age-disease relationship visualization
│
├── report.md                                # Detailed project report
└── README.md                                # This file
```

## Dependencies

### Core Libraries

```python
numpy >= 1.21.5
pandas >= 1.4.4
matplotlib >= 3.6.2
seaborn >= 0.10.1
scikit-learn >= 1.0.2
```

### Explainability Tools

```python
shap >= 0.40.0
pdpbox >= 0.3.0
eli5 >= 0.13.0
```

### Data Analysis Tools

```python
ydata-profiling >= 4.12.0
```

### Visualization Tools

```python
graphviz >= 2.38
pydotplus >= 2.0.2
```

### Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install shap pdpbox eli5
pip install ydata-profiling
pip install graphviz pydotplus
```

**Note**: For Graphviz, you may need to install the system package separately and add it to your PATH.

## Usage

### Running the Complete Pipeline

1. **Environment Setup**: Execute `0_package.ipynb` to install all required packages
2. **Data Exploration**: Run `0_datasetInfo.ipynb` and `1_dataSimpleAnalysisVisualization.ipynb` for EDA
3. **Data Preprocessing**: Execute `2_dataPreprocessing.ipynb` to generate `process_heart.csv`
4. **Model Training**: Run `3_buildRandomForestClassificationModel.ipynb` to train the model
5. **Model Evaluation**: Execute `4_testSetDataPredictionAndModelPerformanceEvaluation.ipynb` for performance metrics
6. **Explainability Analysis**: 
   - Run `6_ShapValue.ipynb` for SHAP value calculation
   - Execute `7_VisualAnalysisOfShapValues.ipynb` for comprehensive SHAP visualizations
   - Run `PermutationImportance.ipynb` for permutation-based importance
   - Execute `pdpbox_0.ipynb` and `pdpbox_1.ipynb` for partial dependence analysis

### Quick Start Example

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load preprocessed data
df = pd.read_csv('process_heart.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

# Train model
model = RandomForestClassifier(
    max_depth=5, 
    n_estimators=100, 
    random_state=5
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
```

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{heart_disease_ml_2024,
  title={Heart Disease Risk Prediction Using Machine Learning: An Explainable AI Approach},
  author={Zhao, Xu},
  year={2024},
  note={EE 475 Machine Learning Project}
}
```

## Dataset Citation

The UCI Heart Disease dataset is publicly available and can be cited as:

```bibtex
@misc{uci_heart_disease,
  title={Heart Disease Dataset},
  author={UCI Machine Learning Repository},
  year={1988},
  url={https://archive.ics.uci.edu/ml/datasets/heart+disease}
}
```

## References

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in neural information processing systems*, 30.

2. Breiman, L. (2001). Random forests. *Machine learning*, 45(1), 5-32.

3. Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of statistics*, 1189-1232.

4. UCI Machine Learning Repository: Heart Disease Dataset. (1988). Retrieved from https://archive.ics.uci.edu/ml/datasets/heart+disease

## License

This project is provided for educational and research purposes. The dataset is publicly available through the UCI Machine Learning Repository.

## Author

**Xu Zhao**  
EE 475 Machine Learning Course Project  
December 2024

## Acknowledgments

- UCI Machine Learning Repository for providing the Heart Disease dataset
- SHAP library developers for explainability tools
- Scikit-learn community for machine learning algorithms

---

**Note**: This project is part of an academic course (EE 475) and is intended for educational purposes. For clinical applications, additional validation and regulatory approval would be required.
