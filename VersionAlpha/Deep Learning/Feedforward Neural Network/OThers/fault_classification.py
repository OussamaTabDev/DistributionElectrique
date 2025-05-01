import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D, Flatten, 
                                    GlobalAveragePooling1D, LSTM, GRU)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import time
import os

# Set random seeds for reproducibility
import tensorflow as tf
import random
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

class FaultClassifier:
    def __init__(self, data_path):
        """
        Initialize the fault classifier with dataset path.
        
        Args:
            data_path (str): Path to the CSV file containing fault data
        """
        self.data_path = data_path
        self.models = {}
        self.histories = {}
        self.evaluation_results = {}
        self.training_times = {}
        self.best_model_name = None
        
        # Fault mapping
        self.fault_map = {
            '0000': 0,  # No Fault
            '1001': 1,  # LG (Line to Ground)
            '0110': 2,  # LL (Line to Line)
            '1011': 3,  # LLG (Line-Line to Ground)
            '0111': 4,  # LLL (Three-phase)
            '1111': 5   # LLLG (Three-phase to Ground)
        }
        
        # Fault names for plotting
        self.fault_names = {
            0: 'No Fault',
            1: 'LG',
            2: 'LL',
            3: 'LLG',
            4: 'LLL',
            5: 'LLLG'
        }
        
    def load_data(self):
        """
        Load and preprocess data from CSV.
        Returns processed train, validation, and test sets.
        """
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        # Create fault type and class
        df['fault_type'] = df[['G', 'C', 'B', 'A']].astype(str).agg(''.join, axis=1)
        df['fault_class'] = df['fault_type'].map(self.fault_map)
        
        # Keep only known fault types
        self.df_unknown = df[~df['fault_type'].isin(self.fault_map.keys())].copy()
        df = df[df['fault_type'].isin(self.fault_map.keys())].copy()
        
        # Split the data into train (60%), validation (20%), and test (20%)
        train, valid, test = np.split(
            df.sample(frac=1, random_state=42), 
            [int(0.6 * len(df)), int(0.8 * len(df))]
        )
        
        print(f"Train set: {len(train)} samples")
        print(f"Validation set: {len(valid)} samples")
        print(f"Test set: {len(test)} samples")
        
        # Process and scale each dataset
        self.X_train, self.Y_train = self._scale_dataset(train)
        self.X_valid, self.Y_valid = self._scale_dataset(valid)
        self.X_test, self.Y_test = self._scale_dataset(test)
        
        # Store original class distribution for analysis
        self.class_distribution = df['fault_class'].value_counts().sort_index()
        print(f"Class distribution: {self.class_distribution.to_dict()}")
        
        return self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.X_test, self.Y_test
    
    def _scale_dataset(self, dataframe):
        """
        Scale features and encode labels from a dataframe.
        
        Args:
            dataframe (pd.DataFrame): Dataframe containing features and labels
            
        Returns:
            tuple: (scaled_features, one_hot_encoded_labels)
        """
        # Extract features and labels
        x = dataframe[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']].values
        y = dataframe['fault_class'].values
        
        # Scale features
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        
        # One-hot encode labels
        y_cat = to_categorical(y, num_classes=len(self.fault_map))
        
        return x_scaled, y_cat
    
    def _reshape_for_model(self, x_data, model_type):
        """
        Reshape input data based on model architecture requirements.
        
        Args:
            x_data (np.array): Input feature array
            model_type (str): Model type ('FNN', 'CNN', 'RNN')
            
        Returns:
            np.array: Reshaped input array
        """
        if model_type in ['CNN', 'LSTM', 'GRU']:
            # Add channel dimension for CNN/RNN models
            return x_data.reshape((x_data.shape[0], x_data.shape[1], 1))
        else:
            # Keep as is for FNN
            return x_data
    
    def build_model(self, model_type, **kwargs):
        """
        Build a neural network model for fault classification.
        
        Args:
            model_type (str): Type of model to build ('FNN', 'CNN', 'LSTM', 'GRU')
            **kwargs: Additional parameters for model customization
            
        Returns:
            keras.Model: Built model
        """
        input_shape = self.X_train.shape[1]
        num_classes = len(self.fault_map)
        
        if model_type == 'FNN':
            model = Sequential([
                Dense(kwargs.get('units_1', 64), activation='relu', input_shape=(input_shape,)),
                Dropout(kwargs.get('dropout_1', 0.2)),
                Dense(kwargs.get('units_2', 32), activation='relu'),
                Dropout(kwargs.get('dropout_2', 0.2)),
                Dense(num_classes, activation='softmax')
            ])
        
        elif model_type == 'ImprovedFNN':
            model = Sequential([
                Dense(kwargs.get('units_1', 128), activation='relu', input_shape=(input_shape,)),
                Dropout(kwargs.get('dropout_1', 0.3)),
                Dense(kwargs.get('units_2', 64), activation='relu'),
                Dropout(kwargs.get('dropout_2', 0.3)),
                Dense(kwargs.get('units_3', 32), activation='relu'),
                Dense(num_classes, activation='softmax')
            ])
        
        elif model_type == 'CNN':
            model = Sequential([
                Conv1D(kwargs.get('filters_1', 64), 
                       kwargs.get('kernel_size', 3), 
                       activation='relu', 
                       input_shape=(input_shape, 1)),
                MaxPooling1D(2),
                Flatten(),
                Dense(kwargs.get('units_1', 32), activation='relu'),
                Dense(num_classes, activation='softmax')
            ])
        
        elif model_type == 'AdvancedCNN':
            model = Sequential([
                Conv1D(kwargs.get('filters_1', 128), 
                       kwargs.get('kernel_size', 3), 
                       activation='relu', 
                       padding='same', 
                       input_shape=(input_shape, 1)),
                MaxPooling1D(2),
                Conv1D(kwargs.get('filters_2', 64), 
                       kwargs.get('kernel_size', 3), 
                       activation='relu', 
                       padding='same'),
                GlobalAveragePooling1D(),
                Dense(num_classes, activation='softmax')
            ])
            
        elif model_type == 'LSTM':
            model = Sequential([
                LSTM(kwargs.get('units_1', 64), input_shape=(input_shape, 1)),
                Dropout(kwargs.get('dropout_1', 0.2)),
                Dense(kwargs.get('units_2', 32), activation='relu'),
                Dense(num_classes, activation='softmax')
            ])
            
        elif model_type == 'GRU':
            model = Sequential([
                GRU(kwargs.get('units_1', 64), input_shape=(input_shape, 1)),
                Dropout(kwargs.get('dropout_1', 0.2)),
                Dense(kwargs.get('units_2', 32), activation='relu'),
                Dense(num_classes, activation='softmax')
            ])
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Compile model
        model.compile(
            optimizer=kwargs.get('optimizer', Adam(learning_rate=kwargs.get('learning_rate', 0.001))),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Built {model_type} model:")
        model.summary()
        
        # Store the model
        self.models[model_type] = model
        return model
    
    def train_model(self, model_type, epochs=30, batch_size=32, use_callbacks=True):
        """
        Train a specified model type.
        
        Args:
            model_type (str): Type of model to train
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            use_callbacks (bool): Whether to use callbacks for training
            
        Returns:
            dict: Training history
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not built yet. Call build_model first.")
        
        model = self.models[model_type]
        
        # Check if we need to reshape input data
        X_train = self._reshape_for_model(self.X_train, model_type)
        X_valid = self._reshape_for_model(self.X_valid, model_type)
        
        # Calculate class weights for imbalanced data
        y_train_classes = np.argmax(self.Y_train, axis=1)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_classes),
            y=y_train_classes
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # Callbacks for better training
        callbacks = []
        if use_callbacks:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
            
            lr_reduction = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
            
            callbacks = [early_stopping, lr_reduction]
        
        print(f"\nTraining {model_type} model...")
        start_time = time.time()
        
        # Train the model
        history = model.fit(
            X_train, self.Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_valid, self.Y_valid),
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        self.training_times[model_type] = training_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Store history
        self.histories[model_type] = history
        return history
    
    def evaluate_model(self, model_type):
        """
        Evaluate a trained model on the test set.
        
        Args:
            model_type (str): Type of model to evaluate
            
        Returns:
            dict: Evaluation metrics
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not built yet. Call build_model first.")
        
        model = self.models[model_type]
        
        # Check if we need to reshape input data
        X_test = self._reshape_for_model(self.X_test, model_type)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(self.Y_test, axis=1)
        
        # Get evaluation metrics
        report = classification_report(y_test_classes, y_pred_classes, output_dict=True)
        conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
        
        # Store results
        self.evaluation_results[model_type] = {
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'accuracy': report['accuracy'],
            'macro_avg_f1': report['macro avg']['f1-score'],
            'weighted_avg_f1': report['weighted avg']['f1-score']
        }
        
        print(f"\nEvaluation of {model_type} model:")
        print(classification_report(y_test_classes, y_pred_classes))
        print("Confusion Matrix:")
        print(conf_matrix)
        
        return self.evaluation_results[model_type]
    
    def compare_models(self):
        """
        Compare all trained models based on evaluation metrics.
        
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        if not self.evaluation_results:
            raise ValueError("No models evaluated yet. Call evaluate_model first.")
        
        # Create comparison dataframe
        comparisons = []
        for model_type, results in self.evaluation_results.items():
            comparisons.append({
                'Model': model_type,
                'Accuracy': results['accuracy'],
                'Macro Avg F1': results['macro_avg_f1'],
                'Weighted Avg F1': results['weighted_avg_f1'],
                'Training Time (s)': self.training_times.get(model_type, float('nan'))
            })
            
        comparison_df = pd.DataFrame(comparisons)
        comparison_df = comparison_df.sort_values('Weighted Avg F1', ascending=False)
        
        # Find best model
        self.best_model_name = comparison_df.iloc[0]['Model']
        
        return comparison_df
    
    def get_best_model(self):
        """
        Return the best performing model based on weighted F1 score.
        
        Returns:
            tuple: (model_name, model)
        """
        if not self.best_model_name:
            self.compare_models()
            
        return self.best_model_name, self.models[self.best_model_name]
    
    def save_best_model(self, output_dir='Deep Learning/Feedforward Neural Network/models'):
        """
        Save the best model to disk.
        
        Args:
            output_dir (str): Directory to save the model
            
        Returns:
            str: Path to saved model
        """
        if not self.best_model_name:
            self.compare_models()
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model
        best_model_path = os.path.join(output_dir, f"{self.best_model_name}.h5")
        save_model(self.models[self.best_model_name], best_model_path)
        
        print(f"Best model ({self.best_model_name}) saved to {best_model_path}")
        return best_model_path
    
    def plot_training_history(self, model_type):
        """
        Plot training history for a specified model.
        
        Args:
            model_type (str): Type of model to plot history for
        """
        if model_type not in self.histories:
            raise ValueError(f"No training history for model {model_type}")
            
        history = self.histories[model_type]
        
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title(f'{model_type} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title(f'{model_type} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'Deep Learning/Feedforward Neural Network/DLFNN/{model_type}_history.png')
        # plt.show()
    
    def plot_confusion_matrix(self, model_type):
        """
        Plot confusion matrix for a specified model.
        
        Args:
            model_type (str): Type of model to plot confusion matrix for
        """
        if model_type not in self.evaluation_results:
            raise ValueError(f"No evaluation results for model {model_type}")
            
        conf_matrix = self.evaluation_results[model_type]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[self.fault_names[i] for i in range(len(self.fault_names))],
                   yticklabels=[self.fault_names[i] for i in range(len(self.fault_names))])
        plt.title(f'{model_type} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'Deep Learning/Feedforward Neural Network/DLFNN/{model_type}_confusion_matrix.png')
        # plt.show()
    
    def plot_class_distribution(self):
        """
        Plot the distribution of classes in the dataset.
        """
        plt.figure(figsize=(10, 6))
        self.class_distribution.plot(kind='bar')
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Fault Class')
        plt.ylabel('Count')
        plt.xticks(ticks=range(len(self.fault_names)), 
                  labels=[self.fault_names[i] for i in range(len(self.fault_names))], 
                  rotation=45)
        plt.tight_layout()
        plt.savefig('Deep Learning/Feedforward Neural Network/DLFNN/class_distribution.png')
        # plt.show()
    
    def plot_model_comparison(self):
        """
        Plot comparison of model performance metrics.
        """
        comparison_df = self.compare_models()
        
        metrics = ['Accuracy', 'Macro Avg F1', 'Weighted Avg F1']
        
        plt.figure(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            plt.subplot(1, 3, i+1)
            sns.barplot(x='Model', y=metric, data=comparison_df)
            plt.title(f'Model Comparison - {metric}')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('Deep Learning/Feedforward Neural Network/model_comparison.png')
        # plt.show()
        
        # Plot training time
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='Training Time (s)', data=comparison_df)
        plt.title('Model Comparison - Training Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('Deep Learning/Feedforward Neural Network/training_time_comparison.png')
        # plt.show()


def main():
    """
    Main function to run the entire workflow.
    """
    # Initialize classifier with data path
    classifier = FaultClassifier('classData.csv')
    
    # Load and preprocess data
    classifier.load_data()
    
    # Plot class distribution
    classifier.plot_class_distribution()
    
    # Build, train, and evaluate different models
    model_types = ['FNN', 'ImprovedFNN', 'CNN', 'AdvancedCNN', 'LSTM', 'GRU']
    
    for model_type in model_types:
        # Build model
        classifier.build_model(model_type)
        
        # Train model
        classifier.train_model(model_type, epochs=30, use_callbacks=True)
        
        # Evaluate model
        classifier.evaluate_model(model_type)
        
        # Plot training history
        classifier.plot_training_history(model_type)
        
        # Plot confusion matrix
        classifier.plot_confusion_matrix(model_type)
    
    # Compare all models
    comparison_df = classifier.compare_models()
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Plot model comparison
    classifier.plot_model_comparison()
    
    # Save best model
    best_model_path = classifier.save_best_model()
    best_model_name, _ = classifier.get_best_model()
    print(f"\nBest model: {best_model_name}")
    print(f"Saved to: {best_model_path}")


if __name__ == "__main__":
    main()