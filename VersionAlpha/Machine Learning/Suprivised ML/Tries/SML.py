import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter

def load_data(filepath):
    """Load and preprocess the fault classification data."""
    df = pd.read_csv(filepath)
    
    # Combine binary outputs (G, C, B, A) into a single fault type label
    df['fault_type'] = df[['G', 'C', 'B', 'A']].astype(str).agg(''.join, axis=1)
    
    return df

def visualize_distributions(df, input_cols):
    """Visualize the distribution of input features for each fault type."""
    unique_faults = df['fault_type'].unique()
    colors = plt.cm.get_cmap('tab10', len(unique_faults))
    
    for in_col in input_cols:
        plt.figure(figsize=(10, 5))
        for i, fault in enumerate(unique_faults):
            subset = df[df['fault_type'] == fault]
            plt.hist(subset[in_col], bins=30, alpha=0.5, density=True,
                     label=f'Fault {fault}', color=colors(i))
        
        plt.title(f"Distribution of {in_col} for each Fault Type")
        plt.xlabel(in_col)
        plt.ylabel("P(X)")
        plt.legend()
        plt.savefig(f"Machine Learning/Suprivised ML/SML/distribution_{in_col}.png")
        plt.close()

def scale_and_encode_dataset(X, y=None):
    """Standardize features and encode class labels."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if y is not None:
        # Convert binary outputs to decimal values (e.g., '1001' -> 9)
        y_decimal = np.zeros(len(y))
        for i in range(len(y)):
            y_decimal[i] = int(''.join(y[i].astype(int).astype(str)), 2)
        
        # Use label encoder to transform non-consecutive integers to consecutive ones
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_decimal)
        
        return X_scaled, y_encoded, label_encoder
    
    return X_scaled
def improve_difficult_classes(X_train, y_train, X_test, y_test, label_encoder):
    """Focus on improving performance for difficult classes (2 and 5)"""
    print("\nApplying techniques to improve classification of difficult classes...")
    
    # Print class distribution before
    print("Class distribution before improvement:")
    print(Counter(y_train))
    
    # 1. Apply SMOTE oversampling - creates synthetic examples of minority classes
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print("Class distribution after SMOTE:")
    print(Counter(y_train_smote))
    
    # 2. Apply class weights to focus more on difficult classes
    # Original class weights would be balanced, but we'll give extra weight to classes 2 and 5
    class_weights = {i: 1 for i in range(len(np.unique(y_train)))}
    class_weights[2] = 3.0  # Triple the importance of class 2
    class_weights[5] = 3.0  # Triple the importance of class 5
    
    # 3. Train models with these improvements
    # Random Forest with class weights and SMOTE data
    rf_weighted = RandomForestClassifier(
        n_estimators=200,  # More trees
        max_depth=15,      # Deeper trees
        min_samples_leaf=1,
        class_weight=class_weights,
        n_jobs=-1,         # Use all CPU cores
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
    rf_result = evaluate_model(rf_weighted, X_test, y_test, "RF (Weighted + SMOTE)", label_encoder)
    plot_confusion_matrix(y_test, rf_result['predictions'], "RF (Weighted + SMOTE)", label_encoder)
    
    print("\nEvaluating XGBoost with weights and SMOTE:")
    xgb_result = evaluate_model(xgb_weighted, X_test, y_test, "XGB (Weighted + SMOTE)", label_encoder)
    plot_confusion_matrix(y_test, xgb_result['predictions'], "XGB (Weighted + SMOTE)", label_encoder)
    
    # 5. Create a specialized ensemble just for the difficult classes
    # This creates a classifier that only focuses on distinguishing between
    # classes 2 and 5 vs the rest
    print("\nTraining specialized classifier for difficult classes...")
    
    # Create binary classification problem: is it class 2 or 5 vs others?
    y_train_binary = np.zeros(len(y_train_smote))
    y_train_binary[np.where((y_train_smote == 2) | (y_train_smote == 5))] = 1
    
    # Train a specialized classifier
    specialized_clf = SVC(kernel='rbf', C=10, gamma=0.01, probability=True)
    specialized_clf.fit(X_train_smote, y_train_binary)
    
    # Then train another classifier only on distinguishing between classes 2 and 5
    # First filter to only class 2 and 5 instances
    mask_2_5 = np.where((y_train_smote == 2) | (y_train_smote == 5))
    X_train_2_5 = X_train_smote[mask_2_5]
    y_train_2_5 = y_train_smote[mask_2_5]
    
    # Make a binary classifier that distinguishes between 2 and 5
    y_train_2_vs_5 = np.zeros(len(y_train_2_5))
    y_train_2_vs_5[np.where(y_train_2_5 == 5)] = 1
    
    clf_2_vs_5 = SVC(kernel='rbf', C=100, gamma=0.1, probability=True)
    clf_2_vs_5.fit(X_train_2_5, y_train_2_vs_5)
    
    # 6. Create a specialized ensemble model
    # This function combines our models to make improved predictions
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
    ensemble_preds = specialized_ensemble_predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, ensemble_preds)
    f1 = f1_score(y_test, ensemble_preds, average='weighted')
    
    print("\n--- Specialized Ensemble Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(classification_report(y_test, ensemble_preds))
    
    # Check specifically how classes 2 and 5 performed
    class_2_indices = np.where(y_test == 2)[0]
    class_5_indices = np.where(y_test == 5)[0]
    
    class_2_accuracy = accuracy_score(y_test[class_2_indices], ensemble_preds[class_2_indices])
    class_5_accuracy = accuracy_score(y_test[class_5_indices], ensemble_preds[class_5_indices])
    
    print(f"Class 2 accuracy: {class_2_accuracy:.4f}")
    print(f"Class 5 accuracy: {class_5_accuracy:.4f}")
    
    return {
        'model_name': 'Specialized Ensemble',
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': ensemble_preds
    }

def split_data(df, input_cols, output_cols, test_size=0.2, valid_size=0.25):
    """Split data into train, validation, and test sets."""
    # First split into train+valid and test
    X = df[input_cols].values
    y = df[output_cols].values
    
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    
    # Then split train+valid into train and valid
    valid_ratio = valid_size / (1 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=valid_ratio, random_state=42)
    
    # Scale the datasets and encode labels
    X_train, y_train, label_encoder = scale_and_encode_dataset(X_train, y_train)
    X_valid, y_valid, _ = scale_and_encode_dataset(X_valid, y_valid)
    X_test, y_test, _ = scale_and_encode_dataset(X_test, y_test)
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test, label_encoder

def evaluate_model(model, X_test, y_test, model_name, label_encoder=None):
    """Evaluate a trained model and return metrics."""
    y_pred = model.predict(X_test)
    
    # If we have a label encoder, we can show the original class labels
    if label_encoder is not None:
        print(f"\nClass mapping: {dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))}")
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n{'-'*50}")
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"{'-'*50}")
    print(classification_report(y_test, y_pred))
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': y_pred
    }

def plot_confusion_matrix(y_test, y_pred, model_name, label_encoder=None):
    """Plot confusion matrix for model evaluation."""
    cm = confusion_matrix(y_test, y_pred)
    
    # Get class labels
    classes = np.unique(np.concatenate((y_test, y_pred)))
    
    # If we have a label encoder, map back to original labels for better interpretation
    if label_encoder is not None:
        class_names = [f"{i} ({label_encoder.classes_[i]})" for i in classes]
    else:
        class_names = classes
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"Machine Learning/Suprivised ML/SML/confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.close()

def compare_models(models_results):
    """Create a bar chart comparing model performances."""
    models = [result['model_name'] for result in models_results]
    accuracies = [result['accuracy'] for result in models_results]
    f1_scores = [result['f1_score'] for result in models_results]
    
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
    plt.savefig("Machine Learning/Suprivised ML/SML/model_comparison.png")
    plt.close()

def train_and_evaluate_models(X_train, X_valid, X_test, y_train, y_valid, y_test, label_encoder):
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
    
    results = []
    
    for name, model in models:
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        result = evaluate_model(model, X_test, y_test, name, label_encoder)
        results.append(result)
        plot_confusion_matrix(y_test, result['predictions'], name, label_encoder)
    
    # Hyperparameter tuning for SVM
    print("\nPerforming grid search for SVM...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }
    
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5)
    grid.fit(X_train, y_train)
    
    print(f"Best SVM parameters: {grid.best_params_}")
    best_svm = grid.best_estimator_
    result = evaluate_model(best_svm, X_test, y_test, 'SVM (Optimized)', label_encoder)
    results.append(result)
    plot_confusion_matrix(y_test, result['predictions'], 'SVM (Optimized)', label_encoder)
    
    # Compare all models
    compare_models(results)
    
    return results, best_svm

def main():
    # Define column names
    input_cols = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
    output_cols = ['G', 'C', 'B', 'A']
    
    # Load data
    filepath = "classData.csv"  # Update with your file path
    df = load_data(filepath)
    
    # Visualize the distributions
    visualize_distributions(df, input_cols)
    
    # Split and scale the data
    X_train, X_valid, X_test, y_train, y_valid, y_test, label_encoder = split_data(
        df, input_cols, output_cols)
    
    # Train and evaluate multiple models
    results, best_model = train_and_evaluate_models(
        X_train, X_valid, X_test, y_train, y_valid, y_test, label_encoder)
    
    ensemble_result = improve_difficult_classes(X_train, y_train, X_test, y_test, label_encoder)
    results.append(ensemble_result)  # Add to your results list for comparison
    # Save the best model and label encoder
    import joblib
    joblib.dump(best_model, 'Machine Learning/Suprivised ML/SML/best_fault_classifier_model.joblib')
    joblib.dump(label_encoder, 'Machine Learning/Suprivised ML/SML/label_encoder.joblib')
    print("\nBest model saved as 'best_fault_classifier_model.joblib'")
    print("Label encoder saved as 'label_encoder.joblib'")
    
    # Print original class mapping
    print("\nClass mapping (encoded -> original):")
    for i, original in enumerate(label_encoder.classes_):
        print(f"  {i} -> {original}")
    
    # Print overall results
    best_result = max(results, key=lambda x: x['f1_score'])
    print(f"\nBest performing model: {best_result['model_name']}")
    print(f"F1 Score: {best_result['f1_score']:.4f}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")

if __name__ == "__main__":
    main()