
        # Model Evaluation Report: Random Forest

        **Generated on**: 2025-05-07 11:40:55

        ## Model Information
        - **Algorithm Type**: RandomForestClassifier
        - **Parameters**: 
        ```python
        {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
        ```

        ## Performance Metrics
        - **Accuracy**: 0.8595
        - **F1 Score (weighted)**: 0.8592
        - **Precision (weighted)**: 0.8602
        - **Recall (weighted)**: 0.8595

        ## Classification Report
        ```text
        0:
  precision: 0.9918
  recall: 0.9979
  f1-score: 0.9948
  support: 482.0000
1:
  precision: 0.9750
  recall: 0.9669
  f1-score: 0.9710
  support: 242.0000
2:
  precision: 0.9950
  recall: 0.9851
  f1-score: 0.9900
  support: 201.0000
3:
  precision: 0.9706
  recall: 0.9706
  f1-score: 0.9706
  support: 238.0000
4:
  precision: 0.4956
  recall: 0.5678
  f1-score: 0.5293
  support: 199.0000
5:
  precision: 0.5191
  recall: 0.4502
  f1-score: 0.4822
  support: 211.0000
accuracy: 0.8595
macro avg:
  precision: 0.8245
  recall: 0.8231
  f1-score: 0.8230
  support: 1573.0000
weighted avg:
  precision: 0.8602
  recall: 0.8595
  f1-score: 0.8592
  support: 1573.0000

        ```

        ## Visualizations

### Confusion Matrix
![Confusion Matrix](Machine Learning/Suprivised ML/FullRapport/visualizations/confusion_matrix_Random_Forest.png)

### Feature Importance
![Feature Importance](Machine Learning/Suprivised ML/FullRapport/visualizations/feature_importance_Random_Forest.png)

### Learning Curve
![Learning Curve](Machine Learning/Suprivised ML/FullRapport/visualizations/learning_curve_Random_Forest.png)

## Recommendations
- This model shows good overall accuracy (>80%).
- Tree-based models like this typically handle complex decision boundaries well.
- They provide feature importance metrics that can help understand the problem.
- Class 4 (5.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 5 (6.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
