import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter
import os
import joblib
from datetime import datetime
from textwrap import dedent

class MLFaultClassifier:
    def __init__(self, data_path="classData.csv"):
        """Initialize the fault classifier with data path and output directories."""
        self.data_path = data_path
        self.input_cols = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
        self.output_cols = ['G', 'C', 'B', 'A']
        self.label_encoder = None
        self.best_model = None
        self.models_results = []
        
        # Create output directories
        self.create_directories()
        
    def create_directories(self):
        """Create directories for outputs if they don't exist."""
        os.makedirs("Machine Learning/Suprivised ML/FullRapport/reports", exist_ok=True)
        os.makedirs("Machine Learning/Suprivised ML/FullRapport/visualizations", exist_ok=True)
        os.makedirs("Machine Learning/Suprivised ML/FullRapport/models", exist_ok=True)
    
    def load_data(self):
        """Load and preprocess the fault classification data."""
        df = pd.read_csv(self.data_path)
        
        # Combine binary outputs (G, C, B, A) into a single fault type label
        df['fault_type'] = df[self.output_cols].astype(str).agg(''.join, axis=1)
        
        return df
    
    def visualize_distributions(self, df):
        """Visualize the distribution of input features for each fault type."""
        unique_faults = df['fault_type'].unique()
        # Fix deprecated get_cmap function
        colors = plt.colormaps['tab10']
        
        for in_col in self.input_cols:
            plt.figure(figsize=(10, 5))
            for i, fault in enumerate(unique_faults):
                subset = df[df['fault_type'] == fault]
                plt.hist(subset[in_col], bins=30, alpha=0.5, density=True,
                         label=f'Fault {fault}', color=colors(i % 10))
            
            plt.title(f"Distribution of {in_col} for each Fault Type")
            plt.xlabel(in_col)
            plt.ylabel("P(X)")
            plt.legend()
            plt.savefig(f"Machine Learning/Suprivised ML/FullRapport/visualizations/distribution_{in_col}.png")
            plt.close()
    
    def scale_and_encode_dataset(self, X, y=None):
        """Standardize features and encode class labels."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if y is not None:
            # Create a mapping from binary string to fault type
            fault_map = {
                '0000': 0,  # No Fault
                '1001': 1,  # LG
                '0011': 2,  # LL
                '0110': 3,  # LL
                '1011': 4,  # LLG
                '0111': 5,  # LLL
                '1111': 6   # LLLG
            }
            
            # Convert binary arrays to string representation and map to fault types
            y_mapped = np.zeros(len(y))
            for i in range(len(y)):
                binary_str = ''.join(y[i].astype(int).astype(str))
                y_mapped[i] = fault_map.get(binary_str, -1)  # -1 for any unmapped patterns
            
            # Use label encoder to ensure consecutive integers
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y_mapped)
            
            return X_scaled, y_encoded
        
        return X_scaled
    
    def split_data(self, df, test_size=0.2, valid_size=0.25):
        """Split data into train, validation, and test sets."""
        # First split into train+valid and test
        X = df[self.input_cols].values
        y = df[self.output_cols].values
        
        X_train_valid, X_test, y_train_valid, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        
        # Then split train+valid into train and valid
        valid_ratio = valid_size / (1 - test_size)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_valid, y_train_valid, test_size=valid_ratio, random_state=42)
        
        # Scale the datasets and encode labels
        X_train, y_train = self.scale_and_encode_dataset(X_train, y_train)
        X_valid, y_valid = self.scale_and_encode_dataset(X_valid, y_valid)
        X_test, y_test = self.scale_and_encode_dataset(X_test, y_test)
        
        return X_train, X_valid, X_test, y_train, y_valid, y_test
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a trained model and return metrics."""
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Generate classification report as dictionary
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        result = {
            'model_name': model_name,
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'predictions': y_pred,
            'true_labels': y_test,  # Add true labels for later use
            'classification_report': report_dict
        }
        
        return result
    
    def plot_confusion_matrix(self, y_test, y_pred, model_name):
        """Plot confusion matrix for model evaluation."""
        cm = confusion_matrix(y_test, y_pred)
        
        # Get class labels
        classes = np.unique(np.concatenate((y_test, y_pred)))
        
        # Map back to original labels for better interpretation
        if self.label_encoder is not None:
            class_names = [f"{i} ({self.label_encoder.classes_[i]})" for i in classes]
        else:
            class_names = classes
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"Machine Learning/Suprivised ML/FullRapport/visualizations/confusion_matrix_{model_name.replace(' ', '_')}.png")
        plt.close()
    
    def plot_feature_importance(self, model, model_name):
        """Plot feature importance for tree-based models."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.figure(figsize=(10, 6))
                plt.title(f"Feature Importances - {model_name}")
                plt.bar(range(len(self.input_cols)), importances[indices],
                       color="r", align="center")
                plt.xticks(range(len(self.input_cols)), 
                          [self.input_cols[i] for i in indices], rotation=45)
                plt.xlim([-1, len(self.input_cols)])
                plt.tight_layout()
                plt.savefig(f"Machine Learning/Suprivised ML/FullRapport/visualizations/feature_importance_{model_name.replace(' ', '_')}.png")
                plt.close()
                return True
        except Exception as e:
            print(f"Could not plot feature importance for {model_name}: {str(e)}")
            return False
    
    def plot_learning_curve(self, model, X_train, y_train, model_name):
        """Plot learning curve for the model."""
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                model, X_train, y_train, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 5))
            
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            
            plt.figure(figsize=(10, 6))
            plt.title(f"Learning Curve - {model_name}")
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            plt.grid()
            
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                           train_scores_mean + train_scores_std, alpha=0.1, color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                           test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                    label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                    label="Cross-validation score")
            
            plt.legend(loc="best")
            plt.savefig(f"Machine Learning/Suprivised ML/FullRapport/visualizations/learning_curve_{model_name.replace(' ', '_')}.png")
            plt.close()
            return True
        except Exception as e:
            print(f"Could not plot learning curve for {model_name}: {str(e)}")
            return False
    
    def plot_decision_tree(self, model, model_name):
        """Plot decision tree for tree-based models."""
        try:
            if hasattr(model, 'tree_'):
                plt.figure(figsize=(20, 10))
                plot_tree(model, filled=True, feature_names=self.input_cols,
                         class_names=[str(c) for c in self.label_encoder.classes_])
                plt.title(f"Decision Tree - {model_name}")
                plt.savefig(f"Machine Learning/Suprivised ML/FullRapport/visualizations/decision_tree_{model_name.replace(' ', '_')}.png")
                plt.close()
                return True
        except Exception as e:
            print(f"Could not plot decision tree for {model_name}: {str(e)}")
            return False
    
    def generate_model_report(self, result):
        """Generate a comprehensive report for a single model."""
        model_name = result['model_name']
        model = result['model']
        report = result['classification_report']
        
        # Format classification report with proper handling of both dict and float values
        report_text = ""
        for key, val in report.items():
            if isinstance(val, dict):
                report_text += f"{key}:\n"
                for metric_name, metric_val in val.items():
                    if isinstance(metric_val, (float, int)):
                        report_text += f"  {metric_name}: {metric_val:.4f}\n"
                    else:
                        report_text += f"  {metric_name}: {metric_val}\n"
            elif isinstance(val, (float, int)):
                report_text += f"{key}: {val:.4f}\n"
            else:
                report_text += f"{key}: {val}\n"
        
        # Create markdown content
        md_content = f"""
        # Model Evaluation Report: {model_name}
        
        **Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        ## Model Information
        - **Algorithm Type**: {model.__class__.__name__}
        - **Parameters**: 
        ```python
        {model.get_params()}
        ```
        
        ## Performance Metrics
        - **Accuracy**: {result['accuracy']:.4f}
        - **F1 Score (weighted)**: {result['f1_score']:.4f}
        - **Precision (weighted)**: {result['precision']:.4f}
        - **Recall (weighted)**: {result['recall']:.4f}
        
        ## Classification Report
        ```text
        {report_text}
        ```
        
        ## Visualizations
        """
        
        # Add visualization references
        md_content += "\n### Confusion Matrix\n"
        md_content += f"![Confusion Matrix](Machine Learning/Suprivised ML/FullRapport/visualizations/confusion_matrix_{model_name.replace(' ', '_')}.png)\n"
        
        if self.plot_feature_importance(model, model_name):
            md_content += "\n### Feature Importance\n"
            md_content += f"![Feature Importance](Machine Learning/Suprivised ML/FullRapport/visualizations/feature_importance_{model_name.replace(' ', '_')}.png)\n"
        
        if self.plot_learning_curve(model, self.X_train, self.y_train, model_name):
            md_content += "\n### Learning Curve\n"
            md_content += f"![Learning Curve](Machine Learning/Suprivised ML/FullRapport/visualizations/learning_curve_{model_name.replace(' ', '_')}.png)\n"
        
        if self.plot_decision_tree(model, model_name):
            md_content += "\n### Decision Tree\n"
            md_content += f"![Decision Tree](Machine Learning/Suprivised ML/FullRapport/visualizations/decision_tree_{model_name.replace(' ', '_')}.png)\n"
        
        # Add model recommendations
        md_content += "\n## Recommendations\n"
        strengths = self.describe_model_strengths(model_name, result)
        md_content += strengths
        
        # Save the report
        report_path = f"Machine Learning/Suprivised ML/FullRapport/reports/{model_name.replace(' ', '_')}_report.md"
        with open(report_path, 'w') as f:
            f.write(dedent(md_content))
        
        print(f"Generated report for {model_name} at {report_path}")
    
    def describe_model_strengths(self, model_name, result):
        """Generate a description of model strengths based on performance."""
        strengths = ""
        
        if result['accuracy'] > 0.9:
            strengths += "- This model shows excellent overall accuracy (>90%).\n"
        elif result['accuracy'] > 0.8:
            strengths += "- This model shows good overall accuracy (>80%).\n"
        else:
            strengths += "- This model's accuracy could be improved.\n"
        
        if model_name.startswith("Random Forest") or model_name.startswith("XGBoost"):
            strengths += "- Tree-based models like this typically handle complex decision boundaries well.\n"
            strengths += "- They provide feature importance metrics that can help understand the problem.\n"
        elif model_name.startswith("SVM"):
            strengths += "- SVM models are effective in high-dimensional spaces.\n"
            strengths += "- They work well with clear margin of separation in the data.\n"
        elif model_name.startswith("Logistic"):
            strengths += "- Linear models are simple and interpretable.\n"
            strengths += "- They work best when the relationship is approximately linear.\n"
        
        # Check for any classes with particularly poor performance
        report = result['classification_report']
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and 'f1-score' in metrics and class_name.isdigit():
                if float(metrics['f1-score']) < 0.7:
                    strengths += f"- Class {class_name} ({self.label_encoder.classes_[int(class_name)]}) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.\n"
        
        return strengths
    
    def compare_models(self):
        """Create comparison visualizations and reports for all models."""
        if not self.models_results:
            print("No models to compare. Train models first.")
            return
        
        # Create radar chart for model comparison
        self.plot_radar_comparison()
        
        # Create bar chart comparison
        self.plot_bar_comparison()
        
        # Generate comprehensive comparison report
        self.generate_comparison_report()
    
    def plot_radar_comparison(self):
        """Create a radar chart comparing multiple metrics across models."""
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        models = list(set([result['model_name'] for result in self.models_results]))
        
        # Number of variables we're plotting
        num_vars = len(metrics)
        
        # Compute angle of each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Complete the loop
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Draw one axe per variable and add labels
        plt.xticks(angles[:-1], metrics)
        
        # Plot each model
        for model in models:
            # Get all metrics for this model
            values = []
            model_results = [r for r in self.models_results if r['model_name'] == model][0]
            for metric in metrics:
                values.append(model_results[metric])
            
            # Complete the loop by appending the first value
            values += values[:1]
            
            # Plot
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend and title
        plt.title('Model Comparison Radar Chart', size=20, pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig("Machine Learning/Suprivised ML/FullRapport/visualizations/model_radar_comparison.png")
        plt.close()
    
    def plot_bar_comparison(self):
        """Create a bar chart comparing model performances."""
        models = [result['model_name'] for result in self.models_results]
        accuracies = [result['accuracy'] for result in self.models_results]
        f1_scores = [result['f1_score'] for result in self.models_results]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        ax.bar(x - width/2, accuracies, width, label='Accuracy')
        ax.bar(x + width/2, f1_scores, width, label='F1 Score')
        
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig("Machine Learning/Suprivised ML/FullRapport/visualizations/model_bar_comparison.png")
        plt.close()
    
    def generate_comparison_report(self):
        """Generate a comprehensive comparison report for all models."""
        md_content = f"""
        # Model Comparison Report
        
        **Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        ## Performance Overview
        """
        
        # Add performance table
        md_content += "\n### Performance Metrics\n"
        md_content += "| Model | Accuracy | F1 Score | Precision | Recall |\n"
        md_content += "|-------|----------|----------|-----------|--------|\n"
        for result in self.models_results:
            md_content += f"| {result['model_name']} | {result['accuracy']:.4f} | {result['f1_score']:.4f} | {result['precision']:.4f} | {result['recall']:.4f} |\n"
        
        # Add visualizations
        md_content += "\n## Visual Comparisons\n"
        md_content += "### Radar Chart Comparison\n"
        md_content += "![Radar Comparison](Machine Learning/Suprivised ML/FullRapport/visualizations/model_radar_comparison.png)\n"
        md_content += "\n### Bar Chart Comparison\n"
        md_content += "![Bar Comparison](Machine Learning/Suprivised ML/FullRapport/visualizations/model_bar_comparison.png)\n"
        
        # Add recommendations
        md_content += "\n## Recommendations\n"
        best_accuracy = max(self.models_results, key=lambda x: x['accuracy'])
        best_f1 = max(self.models_results, key=lambda x: x['f1_score'])
        
        md_content += f"- **For maximum accuracy**, consider using **{best_accuracy['model_name']}** (accuracy: {best_accuracy['accuracy']:.4f})\n"
        md_content += f"- **For balanced performance**, consider using **{best_f1['model_name']}** (F1 score: {best_f1['f1_score']:.4f})\n"
        
        # Save the report
        report_path = "Machine Learning/Suprivised ML/FullRapport/reports/model_comparison_report.md"
        with open(report_path, 'w') as f:
            f.write(dedent(md_content))
        
        print(f"Generated comparison report at {report_path}")
    
    def improve_difficult_classes(self):
        """Focus on improving performance for difficult classes."""
        print("\nApplying techniques to improve classification of difficult classes...")
        
        # Print class distribution before
        print("Class distribution before improvement:")
        print(Counter(self.y_train))
        
        # 1. Apply SMOTE oversampling
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train, self.y_train)
        
        print("Class distribution after SMOTE:")
        print(Counter(y_train_smote))
        
        # 2. Apply class weights to focus more on difficult classes
        class_weights = {i: 1 for i in range(len(np.unique(self.y_train)))}
        class_weights[2] = 3.0  # Triple the importance of class 2
        class_weights[5] = 3.0  # Triple the importance of class 5
        
        # 3. Train models with these improvements
        # Random Forest with class weights and SMOTE data
        rf_weighted = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=1,
            class_weight=class_weights,
            n_jobs=-1,
            random_state=42
        )
        rf_weighted.fit(X_train_smote, y_train_smote)
        
        # XGBoost with class weights and SMOTE data
        xgb_weighted = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=1,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            random_state=42
        )
        
        # XGBoost uses sample_weight parameter instead of class_weight
        sample_weights = np.ones(len(y_train_smote))
        for i, y in enumerate(y_train_smote):
            if y == 2 or y == 5:
                sample_weights[i] = 3.0
        
        xgb_weighted.fit(X_train_smote, y_train_smote, sample_weight=sample_weights)
        
        # 4. Evaluate the models
        print("\nEvaluating Random Forest with weights and SMOTE:")
        rf_result = self.evaluate_model(rf_weighted, self.X_test, self.y_test, "RF (Weighted + SMOTE)")
        self.plot_confusion_matrix(self.y_test, rf_result['predictions'], "RF (Weighted + SMOTE)")
        self.models_results.append(rf_result)
        self.generate_model_report(rf_result)
        
        print("\nEvaluating XGBoost with weights and SMOTE:")
        xgb_result = self.evaluate_model(xgb_weighted, self.X_test, self.y_test, "XGB (Weighted + SMOTE)")
        self.plot_confusion_matrix(self.y_test, xgb_result['predictions'], "XGB (Weighted + SMOTE)")
        self.models_results.append(xgb_result)
        self.generate_model_report(xgb_result)
        
        # 5. Create a specialized ensemble just for the difficult classes
        print("\nTraining specialized classifier for difficult classes...")
        
        # Create binary classification problem: is it class 2 or 5 vs others?
        y_train_binary = np.zeros(len(y_train_smote))
        y_train_binary[np.where((y_train_smote == 2) | (y_train_smote == 5))] = 1
        
        # Train a specialized classifier
        specialized_clf = SVC(kernel='rbf', C=10, gamma=0.01, probability=True)
        specialized_clf.fit(X_train_smote, y_train_binary)
        
        # Then train another classifier only on distinguishing between classes 2 and 5
        mask_2_5 = np.where((y_train_smote == 2) | (y_train_smote == 5))
        X_train_2_5 = X_train_smote[mask_2_5]
        y_train_2_5 = y_train_smote[mask_2_5]
        
        # Make a binary classifier that distinguishes between 2 and 5
        y_train_2_vs_5 = np.zeros(len(y_train_2_5))
        y_train_2_vs_5[np.where(y_train_2_5 == 5)] = 1
        
        clf_2_vs_5 = SVC(kernel='rbf', C=100, gamma=0.1, probability=True)
        clf_2_vs_5.fit(X_train_2_5, y_train_2_vs_5)
        
        # 6. Create a specialized ensemble model
        def specialized_ensemble_predict(X):
            # Get base predictions from the best model (XGBoost)
            base_preds = xgb_weighted.predict(X)
            
            # Get probabilities of being in difficult classes (2 or 5)
            difficult_probs = specialized_clf.predict_proba(X)[:, 1]
            
            # For instances with high probability of being in difficult classes,
            # use the specialized classifier to decide between 2 and 5
            for i in range(len(X)):
                if difficult_probs[i] > 0.6:  # If likely to be class 2 or 5
                    # Predict specifically between 2 and 5
                    if clf_2_vs_5.predict([X[i]])[0] == 0:
                        base_preds[i] = 2  # Class 2
                    else:
                        base_preds[i] = 5  # Class 5
                        
            return base_preds
        
        # Test the ensemble
        ensemble_preds = specialized_ensemble_predict(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, ensemble_preds)
        f1 = f1_score(self.y_test, ensemble_preds, average='weighted')
        precision = precision_score(self.y_test, ensemble_preds, average='weighted')
        recall = recall_score(self.y_test, ensemble_preds, average='weighted')
        
        print("\n--- Specialized Ensemble Results ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (weighted): {f1:.4f}")
        print(classification_report(self.y_test, ensemble_preds))
        
        # Create result dictionary
        ensemble_result = {
            'model_name': 'Specialized Ensemble',
            'model': xgb_weighted,  # Using XGB as the base model for saving
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'predictions': ensemble_preds,
            'true_labels': self.y_test,  # Add true labels for later use
            'classification_report': classification_report(self.y_test, ensemble_preds, output_dict=True)
        }
        
        self.models_results.append(ensemble_result)
        self.plot_confusion_matrix(self.y_test, ensemble_preds, "Specialized Ensemble")
        self.generate_model_report(ensemble_result)
        
        return ensemble_result
    
    def train_and_evaluate_models(self):
        """Train and evaluate multiple models for comparison."""
        models = [
            ('KNN', KNeighborsClassifier(n_neighbors=20)),
            ('Naive Bayes', GaussianNB()),
            ('Logistic Regression', LogisticRegression(max_iter=1000)),
            ('Decision Tree', DecisionTreeClassifier()),
            ('Random Forest', RandomForestClassifier(n_estimators=100)),
            ('SVM (Linear)', SVC(kernel='linear')),
            ('SVM (RBF)', SVC(kernel='rbf')),
            ('SGD Classifier', SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)),
            ('XGBoost', XGBClassifier())
        ]
        
        for name, model in models:
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            result = self.evaluate_model(model, self.X_test, self.y_test, name)
            self.models_results.append(result)
            self.plot_confusion_matrix(self.y_test, result['predictions'], name)
            self.generate_model_report(result)
        
        # Hyperparameter tuning for SVM
        print("\nPerforming grid search for SVM...")
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf']
        }
        
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5)
        grid.fit(self.X_train, self.y_train)
        
        print(f"Best SVM parameters: {grid.best_params_}")
        best_svm = grid.best_estimator_
        result = self.evaluate_model(best_svm, self.X_test, self.y_test, 'SVM (Optimized)')
        self.models_results.append(result)
        self.plot_confusion_matrix(self.y_test, result['predictions'], 'SVM (Optimized)')
        self.generate_model_report(result)
        
        self.best_model = best_svm
        
        # Compare all models
        self.compare_models()
    
    def save_models(self):
        """Save the best model and label encoder."""
        if self.best_model is None:
            print("No model to save. Train models first.")
            return
        
        joblib.dump(self.best_model, 'Machine Learning/Suprivised ML/FullRapport/models/best_fault_classifier_model.joblib')
        joblib.dump(self.label_encoder, 'Machine Learning/Suprivised ML/FullRapport/models/label_encoder.joblib')
        print("\nBest model saved as 'Machine Learning/Suprivised ML/FullRapport/models/best_fault_classifier_model.joblib'")
        print("Label encoder saved as 'Machine Learning/Suprivised ML/FullRapport/models/label_encoder.joblib'")
    
    def run(self):
        """Execute the full pipeline."""
        # Load and prepare data
        df = self.load_data()
        self.visualize_distributions(df)
        self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test = self.split_data(df)
        
        # Print class mapping
        print("\nClass mapping (encoded -> original):")
        for i, original in enumerate(self.label_encoder.classes_):
            print(f"  {i} -> {original}")
        
        # Train and evaluate models
        self.train_and_evaluate_models()
        
        # Improve difficult classes
        self.improve_difficult_classes()
        
        # Save models
        self.save_models()
        
        # Print overall results
        best_result = max(self.models_results, key=lambda x: x['f1_score'])
        print(f"\nBest performing model: {best_result['model_name']}")
        print(f"F1 Score: {best_result['f1_score']:.4f}")
        print(f"Accuracy: {best_result['accuracy']:.4f}")

if __name__ == "__main__":
    classifier = MLFaultClassifier()
    classifier.run()