# shipping-logistics-analytics-ml-dl

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)  

This repository hosts **nine end-to-end logistics analytics projects**, spanning exploratory data analysis, machine learning regression and forecasting, and deep-learning architectures (RNN, CNN, autoencoders). Each folder contains a clear objective, methodology, results, and next steps for real-world shipping and logistics use cases.

**Project Categories:**
- **Data Analysis & Visualization:** EDA, feature engineering, visual storytelling  
- **Machine Learning & Forecasting:** Random forests, gradient boosting, SVR, time-series models  
- **Deep Learning & Computer Vision:** TensorFlow & PyTorch models (RNN, CNN, autoencoders)  

---

## Table of Contents

1. [Summary of Key Model Results](#summary-of-key-model-results)  
2. [Data Analysis & Visualization](#data-analysis--visualization)  
   - [Project 1: Data Preprocessing](#project-1-data-preprocessing)  
   - [Project 2: Visualization & Interpretation](#project-2-visualization--interpretation)  
3. [Machine Learning & Forecasting](#machine-learning--forecasting)  
   - [Project 3: Model Training with scikit-learn](#project-3-model-training-with-scikit-learn)  
   - [Project 6: Time Series Forecasting](#project-6-time-series-forecasting)  
4. [Deep Learning & Computer Vision](#deep-learning--computer-vision)  
   - [Project 4: Neural Network with TensorFlow](#project-4-neural-network-with-tensorflow)  
   - [Project 5: Supply Chain Disruption Prediction (PyTorch)](#project-5-supply-chain-disruption-prediction-pytorch)  
   - [Project 7: Image Classification for Package Inspection](#project-7-image-classification-for-package-inspection)  
   - [Project 8: Anomaly Detection in Shipment Data](#project-8-anomaly-detection-in-shipment-data)  
   - [Project 9: Predictive Maintenance for Fleet Management](#project-9-predictive-maintenance-for-fleet-management)  
5. [License & Contact](#license--contact)  

---

## Summary of Key Model Results

| Project                                         | Metric                  | Business Impact                                            |
|-------------------------------------------------|-------------------------|------------------------------------------------------------|
| Model Training with scikit-learn (LGBMRegressor)| MSE = 0.2532            | ±2-hour ETA error (75% reduction vs. baseline)             |
| Neural Network with TensorFlow                  | MSE = 0.276             | ±2.2-hour ETA error (72% reduction vs. baseline)           |
| Time Series Forecasting (LSTM)                  | MSE = 0.4691            | Multi-step forecast accuracy for capacity planning         |
| Disruption Prediction (PyTorch classifier)      | Accuracy = 82.9%        | Early warning for weather-related shipment delays          |
| Image Classification (CNN)                      | Val. Acc. = 85.9%       | Automated package damage/no-damage sorting                 |
| Anomaly Detection (Autoencoder)                 | Detected 1,229 anomalies| Identified outlier shipments for manual review             |
| Predictive Maintenance (DNN)                    | MSE = 0.8079            | Preliminary RUL estimates for preventive servicing         |

---

## Data Analysis & Visualization

### [Project 1: Data Preprocessing](https://github.com/Chimgozirim/shipment-analytics-prediction-deep-learning/tree/main/Project%201%3A%20Data%20Preprocessing.)  
**Objective:** Clean and preprocess a shipment dataset for modeling.  
**Methods Used:**  
- Loaded 25k-row dataset; verified no missing values or duplicates  
- Selected key features: route, distance, shipping time, weight  
- Encoded categorical routes (LabelEncoder)  
- Normalized numerical features (StandardScaler)  

<details>
<summary>Possible Improvements</summary>

- Use `ColumnTransformer` + `Pipeline` for unified preprocessing  
- Replace LabelEncoder with OneHotEncoder for nominal categories  
- Add correlation heatmap and feature selection step  
- Experiment with feature creation (e.g. weekday, peak hours)

</details>

---

### [Project 2: Visualization & Interpretation](02_visualization_interpretation/README.md)  
**Objective:** Visually explore trends, bottlenecks, and regional performance.  
**Methods Used:**  
- Created time-series plots of daily shipment volumes  
- Mapped average delivery times by region  
- Built dashboard mockups for stakeholder storytelling  

<details>
<summary>Possible Improvements</summary>

- Add interactive Plotly or Power BI dashboards  
- Layer geospatial visualizations with Folium  
- Highlight seasonality and trend decomposition  

</details>

---

## Machine Learning & Forecasting

### [Project 3: Model Training with scikit-learn](03_sklearn_modeling/README.md)  
**Objective:** Train regression models to predict shipment times.  
**Methods Used:**  
- Split data (75% train / 25% test)  
- Trained LinearRegression, RandomForestRegressor, SVR, XGBRegressor, LGBMRegressor  
- Evaluated with Mean Squared Error (MSE)  

**Results:**  
- **LGBMRegressor:** MSE = 0.2532 (best; captures complex interactions)  
- **XGBRegressor:** MSE = 0.2549  
- **RandomForestRegressor:** MSE = 0.2831  
- **SVR:** MSE = 0.3518  
- **LinearRegression:** MSE = 0.7589 (underfits non-linear patterns)  

<details>
<summary>Possible Improvements</summary>

- Hyperparameter tuning with GridSearchCV / Optuna  
- Feature engineering: polynomial features, interaction terms  
- Cross-validation for robust performance estimates  

</details>

---

### [Project 6: Time Series Forecasting](06_time_series_forecasting/README.md)  
**Objective:** Forecast future shipment times using sequence models.  
**Methods Used:**  
- Constructed sliding windows (window=10 timesteps)  
- Built LSTM model (50 units) and trained for 50 epochs  
- Evaluated on rolling multi-step MSE  

**Results:**  
- MSE = 0.4691; demonstrates ability to predict multi-step trends  

<details>
<summary>Possible Improvements</summary>

- Incorporate multivariate inputs (weather, route, volume)  
- Add seasonal-trend decomposition (Prophet or STL)  
- Experiment with Seq2Seq and attention mechanisms  

</details>

---

## Deep Learning & Computer Vision

### [Project 4: Neural Network with TensorFlow](04_tensorflow_nn/README.md)  
**Objective:** Build a feed-forward neural network for shipment-time regression.  
**Methods Used:**  
- Defined 3-layer Sequential model (64 → 32 → 1 neurons)  
- Used Adam optimizer, MSE loss; trained 50 epochs (batch_size=32)  

**Results:**  
- MSE = 0.276; comparable to tree-based models  

<details>
<summary>Possible Improvements</summary>

- Add dropout and batch normalization for regularization  
- Implement early stopping and learning-rate schedules  
- Explore deeper/wider architectures  

</details>

---

### [Project 5: Supply Chain Disruption Prediction (PyTorch)](05_pytorch_disruptions/README.md)  
**Objective:** Classify shipment records as disrupted vs. normal based on weather and external factors.  
**Methods Used:**  
- Merged shipment data with 2012–2022 weather for four cities  
- Created binary `disruption` target from temperature thresholds  
- Built PyTorch classifier (3-layer MLP); trained & evaluated  

**Results:**  
- Accuracy = 82.9% on test set  

<details>
<summary>Possible Improvements</summary>

- Handle class imbalance with weighted loss or resampling  
- Engineer temporal features (lag, rolling statistics)  
- Evaluate with ROC-AUC and F1-score  

</details>

---

### [Project 7: Image Classification for Package Inspection](07_image_classification/README.md)  
**Objective:** Detect damaged vs. intact packages via CNN.  
**Methods Used:**  
- Collected and annotated images (train/val/test) via RoboFlow  
- Applied real-time augmentation (ImageDataGenerator)  
- Built 3-conv-layer CNN; trained 20 epochs  

**Results:**  
- Validation Accuracy = 85.9%  

<details>
<summary>Possible Improvements</summary>

- Apply transfer learning (ResNet, EfficientNet)  
- Add batch normalization and learning-rate scheduling  
- Use Grad-CAM for model explainability  

</details>

---

### [Project 8: Anomaly Detection in Shipment Data](08_anomaly_detection/README.md)  
**Objective:** Identify anomalous shipment records using autoencoders.  
**Methods Used:**  
- Built simple autoencoder with 2-neuron bottleneck  
- Flagged anomalies above 95th-percentile reconstruction error  

**Results:**  
- Detected 1,229 anomalies for manual review  

<details>
<summary>Possible Improvements</summary>

- Upgrade to Variational Autoencoder (VAE)  
- Compare against Isolation Forest and One-Class SVM  
- Tune threshold via precision-recall tradeoff  

</details>

---

### [Project 9: Predictive Maintenance for Fleet Management](09_predictive_maintenance/README.md)  
**Objective:** Predict maintenance needs (remaining useful life) for vehicles.  
**Methods Used:**  
- Preprocessed maintenance logs; selected mileage, age, cost features  
- Built 3-layer DNN (64 → 32 → 1); trained for regression task  

**Results:**  
- MSE = 0.8079 (proof-of-concept stage)  

<details>
<summary>Possible Improvements</summary>

- Incorporate survival analysis (Cox models)  
- Add telematics sensor data (vibration, temperature)  
- Use specialized RUL architectures (DeepSurv, Transformer-based)  

</details>

---

## License & Contact

---
**License:** MIT  
**Contact:** chimgozirimakagha@gmail.com  
