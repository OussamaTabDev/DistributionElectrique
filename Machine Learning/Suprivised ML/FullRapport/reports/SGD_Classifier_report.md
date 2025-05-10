
        # Model Evaluation Report: SGD Classifier

        **Generated on**: 2025-05-07 11:41:56

        ## Model Information
        - **Algorithm Type**: SGDClassifier
        - **Parameters**: 
        ```python
        {'alpha': 0.0001, 'average': False, 'class_weight': None, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'hinge', 'max_iter': 1000, 'n_iter_no_change': 5, 'n_jobs': None, 'penalty': 'l2', 'power_t': 0.5, 'random_state': None, 'shuffle': True, 'tol': 0.001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
        ```

        ## Performance Metrics
        - **Accuracy**: 0.3236
        - **F1 Score (weighted)**: 0.3106
        - **Precision (weighted)**: 0.3367
        - **Recall (weighted)**: 0.3236

        ## Classification Report
        ```text
        0:
  precision: 0.5232
  recall: 0.4917
  f1-score: 0.5070
  support: 482.0000
1:
  precision: 0.5581
  recall: 0.1983
  f1-score: 0.2927
  support: 242.0000
2:
  precision: 0.0000
  recall: 0.0000
  f1-score: 0.0000
  support: 201.0000
3:
  precision: 0.2114
  recall: 0.3908
  f1-score: 0.2743
  support: 238.0000
4:
  precision: 0.1370
  recall: 0.2010
  f1-score: 0.1629
  support: 199.0000
5:
  precision: 0.3074
  recall: 0.4313
  f1-score: 0.3590
  support: 211.0000
accuracy: 0.3236
macro avg:
  precision: 0.2895
  recall: 0.2855
  f1-score: 0.2660
  support: 1573.0000
weighted avg:
  precision: 0.3367
  recall: 0.3236
  f1-score: 0.3106
  support: 1573.0000

        ```

        ## Visualizations

### Confusion Matrix
![Confusion Matrix](Machine Learning/Suprivised ML/FullRapport/visualizations/confusion_matrix_SGD_Classifier.png)

### Learning Curve
![Learning Curve](Machine Learning/Suprivised ML/FullRapport/visualizations/learning_curve_SGD_Classifier.png)

## Recommendations
- This model's accuracy could be improved.
- Class 0 (0.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 1 (1.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 2 (3.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 3 (4.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 4 (5.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 5 (6.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
