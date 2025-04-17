import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib
import re

# Load the dataset
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Data preprocessing function
def preprocess_data(df):
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Extract latitude and longitude from 'Fault Location' column
    if 'Fault Location (Latitude, Longitude)' in data.columns:
        # Parse the coordinates using regex
        # This will handle formats like "(34.0244, -118.728)"
        coordinate_pattern = r'\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)'
        coordinates = data['Fault Location (Latitude, Longitude)'].str.extract(coordinate_pattern)
        
        # Add as new columns
        if not coordinates.empty and coordinates.shape[1] == 2:
            data['Latitude'] = pd.to_numeric(coordinates[0])
            data['Longitude'] = pd.to_numeric(coordinates[1])
        else:
            # If extraction fails, create dummy coordinates to avoid errors
            print("Warning: Could not extract coordinates properly. Creating dummy values.")
            data['Latitude'] = 0
            data['Longitude'] = 0
    
    # Convert categorical features to appropriate types
    categorical_features = ['Fault Type', 'Weather Condition', 'Maintenance Status', 'Component Health']
    for cat_feature in categorical_features:
        if cat_feature in data.columns:
            data[cat_feature] = data[cat_feature].astype('category')
    
    # Handle Fault ID (assuming it's just an identifier and not needed for modeling)
    if 'Fault ID' in data.columns:
        data.drop('Fault ID', axis=1, inplace=True)
    
    # Remove the original location column after extraction
    if 'Fault Location (Latitude, Longitude)' in data.columns:
        data.drop('Fault Location (Latitude, Longitude)', axis=1, inplace=True)
    
    return data

# Feature engineering and preparation for modeling
def prepare_features(data):
    # Identify categorical and numerical columns
    categorical_cols = data.select_dtypes(include=['category', 'object']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Define the features and target variables
    # We'll predict both Fault Type (classification) and Duration of Fault (regression)
    X = data.copy()
    
    y_class = None
    if 'Fault Type' in X.columns:
        y_class = X['Fault Type'].copy()
        X.drop('Fault Type', axis=1, inplace=True)
    
    y_reg = None
    if 'Duration of Fault (hrs)' in X.columns:
        y_reg = X['Duration of Fault (hrs)'].copy()
        X.drop('Duration of Fault (hrs)', axis=1, inplace=True)
    
    # Also drop Down time as it's another target we're not using now
    if 'Down time (hrs)' in X.columns:
        X.drop('Down time (hrs)', axis=1, inplace=True)
    
    # Update categorical and numerical columns based on X
    categorical_cols = [col for col in categorical_cols if col in X.columns]
    numerical_cols = [col for col in numerical_cols if col in X.columns]
    
    print(f"Features being used: {X.columns.tolist()}")
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    return X, y_class, y_reg, categorical_cols, numerical_cols

# Build preprocessing pipeline
def build_preprocessor(categorical_cols, numerical_cols):
    # Create preprocessing steps for categorical and numerical features
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'  # Include columns not specified in transformers
    )
    
    return preprocessor

# Train classification model for Fault Type prediction
def train_classification_model(X, y, categorical_cols, numerical_cols):
    if y is None:
        print("Classification target (Fault Type) not found in the data.")
        return None, None, None
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessing and modeling pipeline
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    
    # Create and train the classification model
    classifier = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    classifier.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = classifier.predict(X_test)
    print("Classification Model Evaluation:")
    print(classification_report(y_test, y_pred))
    
    return classifier, X_test, y_test

# Train regression model for Fault Duration prediction
def train_regression_model(X, y, categorical_cols, numerical_cols):
    if y is None:
        print("Regression target (Duration of Fault) not found in the data.")
        return None, None, None
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessing and modeling pipeline
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    
    # Create and train the regression model
    regressor = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    regressor.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Regression Model Evaluation:\nMSE: {mse:.4f}, R²: {r2:.4f}")
    
    return regressor, X_test, y_test

# Save trained models
def save_models(classifier=None, regressor=None):
    if classifier is not None:
        joblib.dump(classifier, 'fault_type_classifier.pkl')
        print("Classification model saved as 'fault_type_classifier.pkl'")
    
    if regressor is not None:
        joblib.dump(regressor, 'fault_duration_regressor.pkl')
        print("Regression model saved as 'fault_duration_regressor.pkl'")

# Function to make predictions with the saved models
def predict_fault(input_data, classifier=None, regressor=None):
    results = {}
    
    # Convert input_data to DataFrame if it's a dictionary
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    
    # Process the input data similar to training data
    if 'Fault Location (Latitude, Longitude)' in input_data.columns:
        # Extract coordinates
        coordinate_pattern = r'\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)'
        coordinates = input_data['Fault Location (Latitude, Longitude)'].str.extract(coordinate_pattern)
        
        if not coordinates.empty and coordinates.shape[1] == 2:
            input_data['Latitude'] = pd.to_numeric(coordinates[0])
            input_data['Longitude'] = pd.to_numeric(coordinates[1])
        else:
            input_data['Latitude'] = 0
            input_data['Longitude'] = 0
            
        # Drop the original location column
        input_data.drop('Fault Location (Latitude, Longitude)', axis=1, inplace=True)
    
    # Drop unnecessary columns
    if 'Fault ID' in input_data.columns:
        input_data.drop('Fault ID', axis=1, inplace=True)
    if 'Fault Type' in input_data.columns:
        input_data.drop('Fault Type', axis=1, inplace=True)
    if 'Duration of Fault (hrs)' in input_data.columns:
        input_data.drop('Duration of Fault (hrs)', axis=1, inplace=True)
    if 'Down time (hrs)' in input_data.columns:
        input_data.drop('Down time (hrs)', axis=1, inplace=True)
    
    # Print the columns to check if they match what the model expects
    print(f"Input data columns: {input_data.columns.tolist()}")
    
    # Predict fault type
    if classifier is not None:
        try:
            fault_type = classifier.predict(input_data)[0]
            results['Predicted Fault Type'] = fault_type
        except Exception as e:
            print(f"Error predicting fault type: {e}")
    
    # Predict fault duration
    if regressor is not None:
        try:
            duration = regressor.predict(input_data)[0]
            results['Predicted Fault Duration (hrs)'] = duration
        except Exception as e:
            print(f"Error predicting fault duration: {e}")
    
    return results

# Create a simple interface for testing the models
def create_test_interface(classifier=None, regressor=None, sample_data=None):
    if sample_data is None:
        print("No sample data available for testing.")
        return
        
    if classifier is None and regressor is None:
        print("No models available for testing.")
        return
    
    print("\n=== Example Prediction Interface ===")
    
    # Process the sample data first to extract features
    processed_data = preprocess_data(sample_data)
    
    # Get a random sample
    random_sample = processed_data.sample(1).copy()
    
    # Display sample input values
    print("Input values (from a sample record):")
    for key, value in random_sample.iloc[0].items():
        if key not in ['Fault Type', 'Duration of Fault (hrs)', 'Down time (hrs)']:
            print(f"{key}: {value}")
    
    # Prepare input for prediction
    X_sample = random_sample.copy()
    if 'Fault Type' in X_sample.columns:
        X_sample.drop('Fault Type', axis=1, inplace=True)
    if 'Duration of Fault (hrs)' in X_sample.columns:
        X_sample.drop('Duration of Fault (hrs)', axis=1, inplace=True)
    if 'Down time (hrs)' in X_sample.columns:
        X_sample.drop('Down time (hrs)', axis=1, inplace=True)
    
    # Make predictions
    predictions = predict_fault(X_sample, classifier, regressor)
    
    print("\nPrediction Results:")
    for key, value in predictions.items():
        print(f"{key}: {value}")
    
    print("\nTo use the models for your own predictions, you can:")
    print("1. Load the saved models using joblib.load()")
    print("2. Prepare your input data in the same format")
    print("3. Use the predict_fault() function to get predictions")

# Visualize feature importance
def plot_feature_importance(model, categorical_cols, numerical_cols):
    if model is None:
        print("No model provided for feature importance visualization.")
        return
        
    try:
        # Extract feature names correctly based on model structure
        feature_names = []
        
        # Get the preprocessor from the pipeline
        preprocessor = model.named_steps['preprocessor']
        
        # Get names for numerical features (these will be unchanged)
        if len(numerical_cols) > 0:
            feature_names.extend(numerical_cols)
        
        # Get names for categorical features (these will have been one-hot encoded)
        if len(categorical_cols) > 0:
            # Get the encoder for categorical features
            try:
                cat_encoder = preprocessor.named_transformers_['cat']
                cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
                feature_names.extend(cat_feature_names)
            except:
                print("Could not extract categorical feature names.")
        
        # Extract importance
        if 'classifier' in model.named_steps:
            model_step = 'classifier'
        else:
            model_step = 'regressor'
            
        if hasattr(model.named_steps[model_step], 'feature_importances_'):
            importances = model.named_steps[model_step].feature_importances_
            
            # Check if feature_names and importances have the same length
            if len(feature_names) != len(importances):
                print(f"Mismatch in feature names ({len(feature_names)}) and importances ({len(importances)}). Using generic feature names.")
                feature_names = [f"Feature {i}" for i in range(len(importances))]
            
            # Create a dataframe for visualization
            features_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Plot
            plt.figure(figsize=(10, 6))
            top_features = features_df.head(min(10, len(features_df)))
            sns.barplot(x='Importance', y='Feature', data=top_features)
            plt.title('Top Feature Importances')
            plt.tight_layout()
            plt.show()
        else:
            print("Model does not have feature_importances_ attribute.")
    except Exception as e:
        print(f"Error plotting feature importance: {e}")

# Main function to run the entire workflow
def main(file_path=None):
    if file_path is None:
        print("Please provide a file path to your dataset.")
        return
    
    # Load data
    df = load_data(file_path)
    if df is None:
        return
    
    # Display basic info
    print("\nDataset Overview:")
    print(df.info())
    print("\nSample Data:")
    print(df.head())
    
    # Preprocess data
    processed_data = preprocess_data(df)
    print("\nProcessed Data Overview:")
    print(processed_data.columns.tolist())
    
    # Prepare features
    X, y_class, y_reg, categorical_cols, numerical_cols = prepare_features(processed_data)
    
    # Train models
    classifier, X_test_class, y_test_class = None, None, None
    if y_class is not None:
        print("\nTraining classification model...")
        classifier, X_test_class, y_test_class = train_classification_model(X, y_class, categorical_cols, numerical_cols)
    
    regressor, X_test_reg, y_test_reg = None, None, None
    if y_reg is not None:
        print("\nTraining regression model...")
        regressor, X_test_reg, y_test_reg = train_regression_model(X, y_reg, categorical_cols, numerical_cols)
    
    # Plot feature importance
    if classifier is not None:
        print("\nPlotting feature importance for classification model...")
        plot_feature_importance(classifier, categorical_cols, numerical_cols)
    
    # Save models
    print("\nSaving models...")
    save_models(classifier, regressor)
    
    # Create test interface
    print("\nCreating test interface...")
    create_test_interface(classifier, regressor, df)
    
    print("\nProcess completed successfully!")

# Example usage:
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "fault_data.csv"
    main(file_path)

# Function to demonstrate how to use the trained models
def usage_example():
    # Load the trained models
    try:
        classifier = joblib.load('fault_type_classifier.pkl')
        regressor = joblib.load('fault_duration_regressor.pkl')
        print("Models loaded successfully!")
    except FileNotFoundError:
        print("Model files not found. Please run training first.")
        return
    
    # Example input data (must match the format of your training data)
    new_data = {
        'Voltage (V)': 220,
        'Current (A)': 150,
        'Power Load (MW)': 70,
        'Temperature (°C)': 25,
        'Wind Speed (km/h)': 15,
        'Weather Condition': 'Rainy',
        'Maintenance Status': 'Overdue',
        'Component Health': 'Fair',
        'Latitude': 48.8566,
        'Longitude': 2.3522
    }
    
    # Make predictions
    predictions = predict_fault(new_data, classifier, regressor)
    
    print("Prediction for new data:")
    for key, value in predictions.items():
        print(f"{key}: {value}")

# ?------------------------------/?
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.metrics import classification_report, mean_squared_error, r2_score
# import joblib
# import ast

# # Load the dataset (assuming the data is in a CSV file named 'electric_distribution_faults.csv')
# # If your data is in a different format, modify this part accordingly
# def load_data(file_path):
#     try:
#         df = pd.read_csv(file_path)
#         print("Data loaded successfully!")
#         return df
#     except FileNotFoundError:
#         print(f"File not found at {file_path}")
#         return None
#     except Exception as e:
#         print(f"Error loading data: {e}")
#         return None

# # Data preprocessing function
# def preprocess_data(df):
#     # Make a copy to avoid modifying the original dataframe
#     data = df.copy()
    
#     # Extract latitude and longitude from 'Fault Location' column
#     if 'Fault Location (Latitude, Longitude)' in data.columns:
#         data['Fault Location'] = data['Fault Location (Latitude, Longitude)']
#         # Parse the string representation of coordinates into separate columns
#         # This assumes the format is like "(latitude, longitude)"
#         data[['Latitude', 'Longitude']] = data['Fault Location'].str.extract(r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)')
#         data['Latitude'] = pd.to_numeric(data['Latitude'])
#         data['Longitude'] = pd.to_numeric(data['Longitude'])
#         # Drop the original location column
#         data.drop(['Fault Location', 'Fault Location (Latitude, Longitude)'], axis=1, inplace=True, errors='ignore')
    
#     # Convert categorical features to appropriate types
#     categorical_features = ['Fault Type', 'Weather Condition', 'Maintenance Status', 'Component Health']
#     for cat_feature in categorical_features:
#         if cat_feature in data.columns:
#             data[cat_feature] = data[cat_feature].astype('category')
    
#     # Handle Fault ID (assuming it's just an identifier and not needed for modeling)
#     if 'Fault ID' in data.columns:
#         data.drop('Fault ID', axis=1, inplace=True)
    
#     return data

# # Feature engineering and preparation for modeling
# def prepare_features(data):
#     # Identify categorical and numerical columns
#     categorical_cols = data.select_dtypes(include=['category', 'object']).columns.tolist()
#     numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
#     # Define the features and target variables
#     # We'll predict both Fault Type (classification) and Duration of Fault (regression)
#     X = data.drop(['Fault Type', 'Duration of Fault (hrs)', 'Down time (hrs)'], axis=1, errors='ignore')
    
#     y_class = None
#     if 'Fault Type' in data.columns:
#         y_class = data['Fault Type']
    
#     y_reg = None
#     if 'Duration of Fault (hrs)' in data.columns:
#         y_reg = data['Duration of Fault (hrs)']
    
#     # Update categorical and numerical columns based on X
#     categorical_cols = [col for col in categorical_cols if col in X.columns]
#     numerical_cols = [col for col in numerical_cols if col in X.columns]
    
#     return X, y_class, y_reg, categorical_cols, numerical_cols

# # Build preprocessing pipeline
# def build_preprocessor(categorical_cols, numerical_cols):
#     # Create preprocessing steps for categorical and numerical features
#     categorical_transformer = OneHotEncoder(handle_unknown='ignore')
#     numerical_transformer = StandardScaler()
    
#     # Combine preprocessing steps
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numerical_transformer, numerical_cols),
#             ('cat', categorical_transformer, categorical_cols)
#         ])
    
#     return preprocessor

# # Train classification model for Fault Type prediction
# def train_classification_model(X, y, categorical_cols, numerical_cols):
#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Create preprocessing and modeling pipeline
#     preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    
#     # Create and train the classification model
#     classifier = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
#     ])
    
#     classifier.fit(X_train, y_train)
    
#     # Evaluate the model
#     y_pred = classifier.predict(X_test)
#     print("Classification Model Evaluation:")
#     print(classification_report(y_test, y_pred))
    
#     return classifier, X_test, y_test

# # Train regression model for Fault Duration prediction
# def train_regression_model(X, y, categorical_cols, numerical_cols):
#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Create preprocessing and modeling pipeline
#     preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    
#     # Create and train the regression model
#     regressor = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
#     ])
    
#     regressor.fit(X_train, y_train)
    
#     # Evaluate the model
#     y_pred = regressor.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     print(f"Regression Model Evaluation:\nMSE: {mse:.4f}, R²: {r2:.4f}")
    
#     return regressor, X_test, y_test

# # Visualize feature importance
# def plot_feature_importance(model, categorical_cols, numerical_cols):
#     # Extract feature names from the column transformer
#     preprocessor = model.named_steps['preprocessor']
#     cat_encoder = preprocessor.named_transformers_['cat']
#     cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
    
#     # Combine feature names
#     feature_names = list(numerical_cols) + list(cat_feature_names)
    
#     # Extract importance
#     if hasattr(model.named_steps['classifier' if 'classifier' in model.named_steps else 'regressor'], 'feature_importances_'):
#         importances = model.named_steps['classifier' if 'classifier' in model.named_steps else 'regressor'].feature_importances_
        
#         # Create a dataframe for visualization
#         features_df = pd.DataFrame({
#             'Feature': feature_names,
#             'Importance': importances
#         }).sort_values('Importance', ascending=False)
        
#         # Plot
#         plt.figure(figsize=(10, 6))
#         sns.barplot(x='Importance', y='Feature', data=features_df.head(10))
#         plt.title('Top 10 Feature Importances')
#         plt.tight_layout()
#         plt.show()

# # Save trained models
# def save_models(classifier=None, regressor=None):
#     if classifier is not None:
#         joblib.dump(classifier, 'fault_type_classifier.pkl')
#         print("Classification model saved as 'fault_type_classifier.pkl'")
    
#     if regressor is not None:
#         joblib.dump(regressor, 'fault_duration_regressor.pkl')
#         print("Regression model saved as 'fault_duration_regressor.pkl'")

# # Function to make predictions with the saved models
# def predict_fault(input_data, classifier=None, regressor=None):
#     results = {}
    
#     # Convert input_data to DataFrame if it's a dictionary
#     if isinstance(input_data, dict):
#         input_data = pd.DataFrame([input_data])
    
#     # Predict fault type
#     if classifier is not None:
#         fault_type = classifier.predict(input_data)[0]
#         results['Predicted Fault Type'] = fault_type
    
#     # Predict fault duration
#     if regressor is not None:
#         duration = regressor.predict(input_data)[0]
#         results['Predicted Fault Duration (hrs)'] = duration
    
#     return results

# # Create a simple interface for testing the models
# def create_test_interface(classifier=None, regressor=None, sample_data=None):
#     if sample_data is None or classifier is None or regressor is None:
#         print("Models or sample data not available.")
#         return
    
#     # Get a random sample from the dataset to use as an example
#     random_sample = sample_data.sample(1).iloc[0].to_dict()
    
#     print("\n=== Example Prediction Interface ===")
#     print("Input values (from a sample record):")
#     for key, value in random_sample.items():
#         if key not in ['Fault Type', 'Duration of Fault (hrs)', 'Down time (hrs)']:
#             print(f"{key}: {value}")
    
#     # Use the sample for prediction
#     sample_input = pd.DataFrame([random_sample])
#     sample_input = sample_input.drop(['Fault Type', 'Duration of Fault (hrs)', 'Down time (hrs)'], axis=1, errors='ignore')
    
#     # Make predictions
#     predictions = predict_fault(sample_input, classifier, regressor)
    
#     print("\nPrediction Results:")
#     for key, value in predictions.items():
#         print(f"{key}: {value}")
    
#     print("\nTo use the models for your own predictions, you can:")
#     print("1. Load the saved models")
#     print("2. Prepare your input data in the same format")
#     print("3. Use the predict_fault() function to get predictions")

# # Main function to run the entire workflow
# def main(file_path=None):
#     if file_path is None:
#         print("Please provide a file path to your dataset.")
#         return
    
#     # Load data
#     df = load_data(file_path)
#     if df is None:
#         return
    
#     # Display basic info
#     print("\nDataset Overview:")
#     print(df.info())
#     print("\nSample Data:")
#     print(df.head())
    
#     # Preprocess data
#     processed_data = preprocess_data(df)
#     print("\nProcessed Data Overview:")
#     print(processed_data.columns.tolist())
    
#     # Prepare features
#     X, y_class, y_reg, categorical_cols, numerical_cols = prepare_features(processed_data)
    
#     # Train models
#     classifier, X_test_class, y_test_class = None, None, None
#     if y_class is not None:
#         print("\nTraining classification model...")
#         classifier, X_test_class, y_test_class = train_classification_model(X, y_class, categorical_cols, numerical_cols)
    
#     regressor, X_test_reg, y_test_reg = None, None, None
#     if y_reg is not None:
#         print("\nTraining regression model...")
#         regressor, X_test_reg, y_test_reg = train_regression_model(X, y_reg, categorical_cols, numerical_cols)
    
#     # Plot feature importance
#     if classifier is not None:
#         print("\nPlotting feature importance for classification model...")
#         plot_feature_importance(classifier, categorical_cols, numerical_cols)
    
#     # Save models
#     print("\nSaving models...")
#     save_models(classifier, regressor)
    
#     # Create test interface
#     print("\nCreating test interface...")
#     create_test_interface(classifier, regressor, df)
    
#     print("\nProcess completed successfully!")

# # Example usage:
# if __name__ == "__main__":
#     # Replace with your actual file path
#     file_path = "fault_data.csv"
#     main(file_path)

# # How to use the trained models for predictions
# def usage_example():
#     # Load trained models
#     classifier = joblib.load('fault_type_classifier.pkl')
#     regressor = joblib.load('fault_duration_regressor.pkl')
    
#     # Example input data (must match the format of your training data)
#     new_data = {
#         'Voltage (V)': 220,
#         'Current (A)': 150,
#         'Power Load (MW)': 70,
#         'Temperature (°C)': 25,
#         'Wind Speed (km/h)': 15,
#         'Weather Condition': 'Rainy',
#         'Maintenance Status': 'Overdue',
#         'Component Health': 'Fair',
#         'Latitude': 48.8566,
#         'Longitude': 2.3522
#     }
    
#     # Make predictions
#     predictions = predict_fault(new_data, classifier, regressor)
    
#     print("Prediction for new data:")
#     for key, value in predictions.items():
#         print(f"{key}: {value}")