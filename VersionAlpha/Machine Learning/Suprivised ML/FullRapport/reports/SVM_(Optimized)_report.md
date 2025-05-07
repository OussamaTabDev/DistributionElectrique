
        # Model Evaluation Report: SVM (Optimized)

        **Generated on**: 2025-05-07 11:44:35

        ## Model Information
        - **Algorithm Type**: SVC
        - **Parameters**: 
        ```python
        {'C': 100, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 1, 'kernel': 'rbf', 'max_iter': -1, 'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}
        ```

        ## Performance Metrics
        - **Accuracy**: 0.8684
        - **F1 Score (weighted)**: 0.8687
        - **Precision (weighted)**: 0.8705
        - **Recall (weighted)**: 0.8684

        ## Classification Report
        ```text
        0:
  precision: 1.0000
  recall: 1.0000
  f1-score: 1.0000
  support: 482.0000
1:
  precision: 0.9837
  recall: 0.9959
  f1-score: 0.9897
  support: 242.0000
2:
  precision: 1.0000
  recall: 0.9851
  f1-score: 0.9925
  support: 201.0000
3:
  precision: 1.0000
  recall: 0.9790
  f1-score: 0.9894
  support: 238.0000
4:
  precision: 0.5000
  recall: 0.5829
  f1-score: 0.5383
  support: 199.0000
5:
  precision: 0.5246
  recall: 0.4550
  f1-score: 0.4873
  support: 211.0000
accuracy: 0.8684
macro avg:
  precision: 0.8347
  recall: 0.8330
  f1-score: 0.8329
  support: 1573.0000
weighted avg:
  precision: 0.8705
  recall: 0.8684
  f1-score: 0.8687
  support: 1573.0000

        ```

        ## Visualizations

### Confusion Matrix
![Confusion Matrix](Machine Learning/Suprivised ML/FullRapport/visualizations/confusion_matrix_SVM_(Optimized).png)

### Learning Curve
![Learning Curve](Machine Learning/Suprivised ML/FullRapport/visualizations/learning_curve_SVM_(Optimized).png)

## Recommendations
- This model shows good overall accuracy (>80%).
- SVM models are effective in high-dimensional spaces.
- They work well with clear margin of separation in the data.
- Class 4 (5.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 5 (6.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
