# Steel-Plate-Fault-Classification-Using-Machine-Learning
A Pattern Recognition project focused on classifying steel plate surface defects into seven categories using feature engineering, multiple classification algorithms, and a weighted soft-voting ensemble. Models are evaluated using stratified 10-fold cross-validation and performance metrics such as Accuracy, Macro-F1, and Weighted-F1.
# Steel Plate Fault Classification Using Machine Learning

<p align="center">
  <b>Pattern Recognition | Machine Learning | Supervised Classification</b>
</p>

## 📌 About The Project

Steel plate surface defects can affect product quality and manufacturing reliability. Identifying these defects accurately is an important task in industrial quality inspection.

This project implements a machine learning-based approach to classify steel plate surface defects into seven different categories using pattern recognition techniques.

The project involves data preprocessing, feature engineering, implementation of multiple machine learning classifiers, model comparison, and ensemble learning. Different models are evaluated using Stratified 10-Fold Cross-Validation to identify their classification performance.

## 🎯 Objectives

- Develop a machine learning model to classify steel plate surface defects.
- Perform data preprocessing and feature engineering.
- Implement and compare multiple classification algorithms.
- Evaluate models using Accuracy, Macro-F1, and Weighted-F1 scores.
- Implement a weighted soft-voting ensemble classifier.
- Analyze classification performance using confusion matrices and visualizations.

## 📂 Dataset Description

The project uses the **Steel Plates Faults Dataset** from the UCI Machine Learning Repository.

- Total labelled samples: 1,941
- Original input features: 27
- Classification type: Supervised Learning
- Number of target classes: 7

### Defect Categories

1. Pastry
2. Z_Scratch
3. K_Scatch
4. Stains
5. Dirtiness
6. Bumps
7. Other_Faults

## ⚙️ Methodology

The project follows these steps:

**1. Data Preprocessing**
- Load and explore the dataset.
- Remove samples without positive target indicators.
- Prepare target labels for seven-class classification.
- Apply stratified data splitting for cross-validation.

**2. Feature Engineering**

Four additional features are constructed:
- Aspect Ratio
- Shape Compactness
- Thickness–Perimeter Ratio
- Luminosity Spread

**3. Model Implementation**

Eleven classifier configurations are implemented and evaluated.

**4. Ensemble Learning**

A weighted soft-voting ensemble is developed using:
- XGBoost
- LightGBM
- CatBoost
- Extra Trees

**5. Model Evaluation**

Stratified 10-Fold Cross-Validation is used to evaluate the models based on:
- Accuracy
- Macro-F1 Score
- Weighted-F1 Score
- Precision and Recall
- Confusion Matrix

## 🤖 Machine Learning Algorithms

The following algorithms are compared:

| Algorithm | Type |
|---|---|
| Logistic Regression | Linear Classifier |
| RBF SVM | Support Vector Machine |
| K-Nearest Neighbors | Instance-Based Learning |
| Decision Tree | Tree-Based Learning |
| Random Forest | Ensemble Learning |
| Extra Trees | Ensemble Learning |
| Multi-Layer Perceptron | Neural Network |
| XGBoost | Gradient Boosting |
| LightGBM | Gradient Boosting |
| CatBoost | Gradient Boosting |
| Weighted Soft-Voting | Ensemble Classifier |

## 📊 Results and Performance Analysis

The performance of all classifiers is evaluated using Stratified 10-Fold Cross-Validation.

### Model Performance Comparison

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| LightGBM | 81.09% | 82.73% |
| Soft-Voting Ensemble | 80.89% | 82.73% |
| XGBoost | 80.47% | 82.85% |
| CatBoost | 79.08% | 81.20% |
| Random Forest | 77.69% | 79.58% |
| Extra Trees | 77.48% | 79.48% |
| RBF SVM | 76.76% | 78.53% |
| MLP | 76.45% | 77.59% |
| KNN | 74.75% | 76.73% |
| Decision Tree | 73.62% | 74.45% |
| Logistic Regression | 71.51% | 70.78% |

### Key Findings

- LightGBM achieved the highest mean accuracy of **81.09%**.
- XGBoost achieved the highest Macro-F1 score of **82.85%**.
- The proposed soft-voting ensemble achieved **80.89% accuracy**.
- Pastry, Dirtiness, and Bumps were comparatively difficult defect categories to classify.
- Feature engineering provided improvements for some models, particularly MLP and the ensemble.

## 📈 Visualizations

The project includes the following performance visualizations:

- Classifier Accuracy Comparison
- Feature Engineering Ablation Study
- Fold-Wise Performance Stability
- Class-Wise Precision, Recall and F1-Score
- Normalized Confusion Matrix

These visualizations help analyze the performance, stability, and classification errors of the implemented models.

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Machine Learning
- Pattern Recognition

## 📚 References

1. Steel Plates Faults Dataset – UCI Machine Learning Repository  
   https://doi.org/10.24432/C5J88N

2. Scikit-learn Documentation  
   https://scikit-learn.org/

3. XGBoost Documentation  
   https://xgboost.readthedocs.io/

4. LightGBM Documentation  
   https://lightgbm.readthedocs.io/

5. CatBoost Documentation  
   https://catboost.ai/

## 👨‍💻 Project Team

**Yelakala Jignas**  
**Usarthi Hemanth Kumar**

Department of Electronics and Communication Engineering  
Indian Institute of Information Technology, Sri City

**Course:** Pattern Recognition  
**Academic Year:** 2026–2027

---

⭐ If you find this project useful, consider giving the repository a star!
