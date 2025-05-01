"""
Script for using the trained model to make predictions on new data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

class FaultPredictor:
    def __init__(self, model_path):
        """
        Initialize the fault predictor with a trained model.
        
        Args:
            model_path (str): Path to the saved model file (.h5)
        """
        self.model = load_model(model_path)
        self.fault_map = {
            0: 'No Fault',
            1: 'LG (Line to Ground)',
            2: 'LL (Line to Line)',
            3: 'LLG (Line-Line to Ground)',
            4: 'LLL (Three-phase)',
            5: 'LLLG (Three-phase to Ground)'
        }
        self.scaler = StandardScaler()
        
    def preprocess_data(self, data):
        """
        Preprocess input data for prediction.
        
        Args:
            data (pd.DataFrame or np.array): Input data with features 
                                            ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
        
        Returns:
            np.array: Preprocessed data ready for model prediction
        """
        # If input is a DataFrame, extract relevant features
        if isinstance(data, pd.DataFrame):
            features = data[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']].values
        else:
            features = np.array(data)
            
        # Scale the features
        scaled_features = self.scaler.fit_transform(features)
        
        # Check if the model expects 3D input (for CNN, LSTM, GRU)
        if len(self.model.input_shape) > 2:
            # Reshape for CNN/RNN models
            scaled_features = scaled_features.reshape((scaled_features.shape[0], 
                                                      scaled_features.shape[1], 1))
            
        return scaled_features
        
    def predict(self, data):
        """
        Make fault predictions on input data.
        
        Args:
            data (pd.DataFrame or np.array): Input data with features 
                                            ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
        
        Returns:
            tuple: (predicted_class_indices, predicted_probabilities, predicted_class_names)
        """
        # Preprocess the data
        processed_data = self.preprocess_data(data)
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        
        # Get class indices and probabilities
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Map to class names
        predicted_names = [self.fault_map[cls] for cls in predicted_classes]
        
        return predicted_classes, predictions, predicted_names
    
    def visualize_prediction(self, data, index=0):
        """
        Visualize the prediction probabilities for a single sample.
        
        Args:
            data (pd.DataFrame or np.array): Input data
            index (int): Index of the sample to visualize
        """
        _, predictions, _ = self.predict(data)
        
        # Get probabilities for the specified sample
        probs = predictions[index]
        
        plt.figure(figsize=(10, 6))
        plt.bar(self.fault_map.values(), probs)
        plt.title('Fault Classification Probabilities')
        plt.xlabel('Fault Type')
        plt.ylabel('Probability')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def main():
    """
    Example usage of the FaultPredictor
    """
    # Load the model
    predictor = FaultPredictor('Deep Learning/Feedforward Neural Network/models/ImprovedFNN.h5')  # Replace with your best model
    
    # Example: Read some test data
    try:
        test_data = pd.read_csv('classData.csv')  # Replace with your test data path
        
        # Make predictions
        class_indices, probabilities, class_names = predictor.predict(test_data)
        
        # Print predictions for the first few samples
        for i in range(min(5, len(test_data))):
            print(f"Sample {i+1}:")
            print(f"  Predicted class: {class_names[i]}")
            print(f"  Probability: {probabilities[i][class_indices[i]]:.4f}")
            print()
            
        # Visualize one prediction
        predictor.visualize_prediction(test_data, index=0)
        
    except FileNotFoundError:
        print("Test data file not found. Please provide a valid CSV file with the required features.")
        print("Required features: ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']")


if __name__ == "__main__":
    main()