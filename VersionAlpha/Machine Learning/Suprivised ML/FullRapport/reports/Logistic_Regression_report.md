
        # Model Evaluation Report: Logistic Regression

        **Generated on**: 2025-05-07 11:40:04

        ## Model Information
        - **Algorithm Type**: LogisticRegression
        - **Parameters**: 
        ```python
        {'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 1000, 'multi_class': 'deprecated', 'n_jobs': None, 'penalty': 'l2', 'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
        ```

        ## Performance Metrics
        - **Accuracy**: 0.3719
        - **F1 Score (weighted)**: 0.2525
        - **Precision (weighted)**: 0.3041
        - **Recall (weighted)**: 0.3719

        ## Classification Report
        ```text
        0:
  precision: 0.3495
  recall: 1.0000
  f1-score: 0.5180
  support: 482.0000
1:
  precision: 0.0000
  recall: 0.0000
  f1-score: 0.0000
  support: 242.0000
2:
  precision: 0.0000
  recall: 0.0000
  f1-score: 0.0000
  support: 201.0000
3:
  precision: 1.0000
  recall: 0.2353
  f1-score: 0.3810
  support: 238.0000
4:
  precision: 0.0000
  recall: 0.0000
  f1-score: 0.0000
  support: 199.0000
5:
  precision: 0.3406
  recall: 0.2227
  f1-score: 0.2693
  support: 211.0000
accuracy: 0.3719
macro avg:
  precision: 0.2817
  recall: 0.2430
  f1-score: 0.1947
  support: 1573.0000
weighted avg:
  precision: 0.3041
  recall: 0.3719
  f1-score: 0.2525
  support: 1573.0000

        ```

        ## Visualizations

### Confusion Matrix
![Confusion Matrix](Machine Learning/Suprivised ML/FullRapport/visualizations/confusion_matrix_Logistic_Regression.png)

### Learning Curve
![Learning Curve](Machine Learning/Suprivised ML/FullRapport/visualizations/learning_curve_Logistic_Regression.png)

## Recommendations
- This model's accuracy could be improved.
- Linear models are simple and interpretable.
- They work best when the relationship is approximately linear.
- Class 0 (0.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 1 (1.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 2 (3.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 3 (4.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 4 (5.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 5 (6.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
