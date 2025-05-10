
        # Model Evaluation Report: Naive Bayes

        **Generated on**: 2025-05-07 11:40:01

        ## Model Information
        - **Algorithm Type**: GaussianNB
        - **Parameters**: 
        ```python
        {'priors': None, 'var_smoothing': 1e-09}
        ```

        ## Performance Metrics
        - **Accuracy**: 0.8010
        - **F1 Score (weighted)**: 0.7598
        - **Precision (weighted)**: 0.7974
        - **Recall (weighted)**: 0.8010

        ## Classification Report
        ```text
        0:
  precision: 0.9377
  recall: 1.0000
  f1-score: 0.9679
  support: 482.0000
1:
  precision: 0.8622
  recall: 0.9050
  f1-score: 0.8831
  support: 242.0000
2:
  precision: 0.9163
  recall: 0.9254
  f1-score: 0.9208
  support: 201.0000
3:
  precision: 0.8313
  recall: 0.8487
  f1-score: 0.8399
  support: 238.0000
4:
  precision: 0.4743
  recall: 0.8342
  f1-score: 0.6047
  support: 199.0000
5:
  precision: 0.5556
  recall: 0.0237
  f1-score: 0.0455
  support: 211.0000
accuracy: 0.8010
macro avg:
  precision: 0.7629
  recall: 0.7562
  f1-score: 0.7103
  support: 1573.0000
weighted avg:
  precision: 0.7974
  recall: 0.8010
  f1-score: 0.7598
  support: 1573.0000

        ```

        ## Visualizations

### Confusion Matrix
![Confusion Matrix](Machine Learning/Suprivised ML/FullRapport/visualizations/confusion_matrix_Naive_Bayes.png)

### Learning Curve
![Learning Curve](Machine Learning/Suprivised ML/FullRapport/visualizations/learning_curve_Naive_Bayes.png)

## Recommendations
- This model shows good overall accuracy (>80%).
- Class 4 (5.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 5 (6.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
