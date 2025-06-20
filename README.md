# Pumpkin Seed Classification Using Ensemble Learning

This project focuses on classifying pumpkin seed varieties using various machine learning models, and finally stacking their outputs using CatBoost for improved performance.

---

## Step-by-Step Procedure

### 1. **Libraries Used**
- pandas, numpy: for data manipulation
- matplotlib, seaborn: for visualization
- scikit-learn: preprocessing, model training, evaluation
- catboost, lightgbm, xgboost: gradient boosting frameworks

---

### 2. **Data Processing**
- Target: Class (categorical, 2 classes)
- Applied label encoding to target
- Standardized features using StandardScaler
- Split into X_train, X_test, y_train, y_test (60/40)
- Generated out-of-fold predictions using K-Fold Cross Validation

---

##  Models Used

A total of **11 models** were used as base classifiers:
Model Performance Summary (on Train Set)
| No. | Model                          | Precision-Recall (%) | ROC AUC (%) | Accuracy (%) |
|-----|--------------------------------|---------------|-------------|---------------|
| 1   | Logistic Regression            | 94          | 94       | 88.4         |
| 2   | Random Forest Classifier       | 94         | 94        | 89.3          |
| 3   | K-Nearest Neighbors (KNN)      | 93          | 93        | 87.6          |
| 4   | Support Vector Classifier      | 95         | 94        | 89          |
| 5   | Decision Tree Classifier       | 86          | 89        | 84          |
| 6   | Gaussian Naive Bayes (GNB)     | 94          | 94        | 87.25          |
| 7   | Linear Discriminant Analysis   | 95          | 94        | 88.1         |
| 8   | Multi-layer Perceptron (MLP)   | 95         | 95        | 88.55          |
| 9   | LightGBM                       | 95          | 95        | 87.8          |
| 10  | XGBoost                        | 94          | 94        | 87.5          |
| 11  | CatBoost                       | 96          | 95       | 88.2          |

----
Model Performance Summary (on Test Set)
| No. | Model                          | Precision-Recall (%) | ROC AUC (%) | Accuracy (%) |
|-----|--------------------------------|---------------|-------------|---------------|
| 1   | Logistic Regression            | 95          | 93       | 85.4         |
| 2   | Random Forest Classifier       | 95         | 93        | 87.4          |
| 3   | K-Nearest Neighbors (KNN)      | 93          | 93        | 83.4          |
| 4   | Support Vector Classifier      | 95         | 93        | 86.4          |
| 5   | Decision Tree Classifier       | 85          | 93        | 82.6          |
| 6   | Gaussian Naive Bayes (GNB)     | 94          | 93        | 86.2          |
| 7   | Linear Discriminant Analysis   | 93          | 93        | 87         |
| 8   | Multi-layer Perceptron (MLP)   | 95         | 95        |  87         |
| 9   | LightGBM                       | 95          | 95        | 86.6          |
| 10  | XGBoost                        | 95          | 94        | 87.2          |
| 11  | CatBoost                       | 95          | 95       | 86.8          |

---

## Final Model: CatBoost (Stacked Ensemble)

- 8 model predictions were combined with original features.
- CatBoostClassifier was used as a meta-model.
- Trained using full training set and evaluated on test set.
  
**Ensemble Results(Train set):**
- **Precision-Recall:** 95%
- **ROC AUC:** 95%
- **accuracy:** 96.5%

**Ensemble Results(Test set):**
- **Precision-Recall:** 100%
- **ROC AUC:** 95%
- **accuracy:** 99%
---

##  Conclusion

Using a stacking ensemble of 8 diverse models with **CatBoost** as the final learner provided excellent predictive power on the pumpkin seed dataset, achieving:
- **100% accuracy**
- **95% ROC AUC**
- **100% Precision-Recall** 
This workflow demonstrates the effectiveness of model stacking combined with a high-performing gradient booster like CatBoost.
