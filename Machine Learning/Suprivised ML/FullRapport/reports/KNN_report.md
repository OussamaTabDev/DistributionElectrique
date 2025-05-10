
        # Model Evaluation Report: KNN

        **Generated on**: 2025-05-07 11:39:40

        ## Model Information
        - **Algorithm Type**: KNeighborsClassifier
        - **Parameters**: 
        ```python
        {'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 20, 'p': 2, 'weights': 'uniform'}
        ```

        ## Performance Metrics
        - **Accuracy**: 0.8411
        - **F1 Score (weighted)**: 0.8339
        - **Precision (weighted)**: 0.8343
        - **Recall (weighted)**: 0.8411

        ## Classification Report
        ```text
        0:
  precision: 0.9817
  recall: 1.0000
  f1-score: 0.9908
  support: 482.0000
1:
  precision: 0.9255
  recall: 0.9752
  f1-score: 0.9497
  support: 242.0000
2:
  precision: 1.0000
  recall: 0.9701
  f1-score: 0.9848
  support: 201.0000
3:
  precision: 0.9136
  recall: 0.9328
  f1-score: 0.9231
  support: 238.0000
4:
  precision: 0.4902
  recall: 0.6281
  f1-score: 0.5507
  support: 199.0000
5:
  precision: 0.4701
  recall: 0.2986
  f1-score: 0.3652
  support: 211.0000
accuracy: 0.8411
macro avg:
  precision: 0.7968
  recall: 0.8008
  f1-score: 0.7940
  support: 1573.0000
weighted avg:
  precision: 0.8343
  recall: 0.8411
  f1-score: 0.8339
  support: 1573.0000

        ```

        ## Visualizations

### Confusion Matrix
![Confusion Matrix](Machine Learning/Suprivised ML/FullRapport/visualizations/confusion_matrix_KNN.png)

### Learning Curve
![Learning Curve](Machine Learning/Suprivised ML/FullRapport/visualizations/learning_curve_KNN.png)

## Recommendations
- This model shows good overall accuracy (>80%).
- Class 4 (5.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 5 (6.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
