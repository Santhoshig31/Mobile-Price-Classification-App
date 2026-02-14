# Mobile Price Classification - Machine Learning Assignment 2

## 1. Problem Statement
The objective of this project is to build a Machine Learning model to classify mobile phones into one of four price ranges (Low, Medium, High, Very High) based on their specifications such as RAM, Battery Power, Camera quality, etc. This is a multi-class classification problem aimed at assisting in automated price estimation.

## 2. Dataset Description
* **Source**: Kaggle (Mobile Price Classification)
* **Type**: Classification (Multi-class)
* **Rows**: 2000
* **Columns**: 21
* **Target Variable**: `price_range` (0: Low Cost, 1: Medium Cost, 2: High Cost, 3: Very High Cost)
* **Features**: Includes numeric features like `ram`, `battery_power`, `px_height`, `px_width`, `int_memory`, etc.

## 3. Models Implemented
The following 6 Machine Learning models were implemented and evaluated:
1.  Logistic Regression
2.  Decision Tree Classifier
3.  K-Nearest Neighbors (KNN)
4.  Naive Bayes (Gaussian)
5.  Random Forest Classifier (Ensemble)
6.  XGBoost Classifier (Ensemble)

## 4. Model Comparison Table
| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.975 | 1.000 | 0.976 | 0.975 | 0.975 | 0.967 |
| **Decision Tree** | 0.833 | 0.886 | 0.834 | 0.833 | 0.832 | 0.777 |
| **KNN** | 0.530 | 0.763 | 0.570 | 0.530 | 0.541 | 0.379 |
| **Naive Bayes** | 0.797 | 0.956 | 0.806 | 0.797 | 0.799 | 0.731 |
| **Random Forest** | 0.892 | 0.983 | 0.896 | 0.892 | 0.893 | 0.857 |
| **XGBoost** | 0.905 | 0.991 | 0.906 | 0.905 | 0.905 | 0.874 |

## 5. Observations
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | **Best Performing Model (97.5% Accuracy).** It achieved a perfect AUC of 1.000. This indicates that the boundaries between price ranges in this dataset are strictly linear (likely defined by hard thresholds in RAM or Battery Power). |
| **Decision Tree** | Achieved moderate accuracy (83.3%). While easy to interpret, single trees tend to overfit the training data and struggle to generalize as well as ensemble methods or linear models on this specific clean dataset. |
| **KNN** | **Lowest Performance (53.0%).** Even with scaling, KNN struggled. This suggests that the data points for different price ranges overlap significantly in the high-dimensional space, making distance-based classification difficult with standard k=5. |
| **Naive Bayes** | performed reasonably well (79.7%) but lagged behind ensemble methods. The assumption of feature independence (e.g., that RAM and Battery are unrelated) may not fully hold true here. |
| **Random Forest** | Strong performance (89.2%) with a high AUC (0.983). It significantly improved upon the single Decision Tree by averaging multiple trees, reducing variance and overfitting. |
| **XGBoost** | Second best model (90.5%). It effectively captured complex patterns using gradient boosting. However, for this specific dataset, the simpler Logistic Regression outperformed it, proving that complex models aren't always better for linear problems. |

## 6. Project Structure
* `app.py`: Streamlit application file (Main interface).
* `models/`: Folder containing trained model files (`.pkl`) and the scaler.
* `requirements.txt`: List of python dependencies for deployment.
* `mobile_price_data.csv`: Dataset used for training.