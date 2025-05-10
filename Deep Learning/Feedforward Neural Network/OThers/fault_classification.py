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
from tensorflow.keras.utils import to_categorical, plot_model
import time
import os
import json
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from datetime import datetime
import squarify  # For tree maps 

# Set random seeds for reproducibility
import tensorflow as tf
import random
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

class EnhancedFaultClassifier:
    def __init__(self, data_path):
        """
        Initialize the enhanced fault classifier with dataset path.
        
        Args:
            data_path (str): Path to the CSV file containing fault data
        """
        self.data_path = data_path
        self.models = {}
        self.histories = {}
        self.evaluation_results = {}
        self.training_times = {}
        self.best_model_name = None
        self.model_params = {}
        self.model_layer_counts = {}
        self.model_memory_usage = {}
        self.prediction_speeds = {}
        self.output_dir = 'Fault_Classification_Results'
        
        # Create output directories
        self.create_output_dirs()
        
        # Fault mapping
        self.fault_map = {
            '0000': 0,  # No Fault
            '1001': 1,  # LG
            '0011': 2,  # LL AB
            '0110': 3,  # LL BC
            '1011': 4,  # LLG
            '0111': 5,  # LLL
            '1111': 6   # LLLG
        }
        
        # Fault names for plotting
        self.fault_names = {
            0: 'No Fault',
            1: 'LG',
            2: 'LL AB',
            3: 'LL BC',
            4: 'LLG',
            5: 'LLL',
            6: 'LLLG'
        }
        
        # Model architecture descriptions
        self.model_descriptions = {
            'FNN': 'Standard Feedforward Neural Network with two hidden layers',
            'ImprovedFNN': 'Enhanced Feedforward Neural Network with three hidden layers and increased complexity',
            'CNN': 'Convolutional Neural Network with single 1D convolution layer',
            'AdvancedCNN': 'Complex CNN with multiple convolutional layers and global average pooling',
            'LSTM': 'Long Short-Term Memory recurrent network for sequence processing',
            'GRU': 'Gated Recurrent Unit network, a more efficient variant of LSTM'
        }
        
        # Extended model descriptions for detailed information
        self.extended_descriptions = {
            'FNN': """
                ## Feedforward Neural Network (FNN)
                
                ### Architecture Overview
                The FNN is the simplest deep learning architecture, consisting of fully connected layers where each neuron connects to every neuron in the previous and next layer.
                
                ### Key Features
                - **Structure**: Input layer → Hidden layer 1 (64 neurons) → Hidden layer 2 (32 neurons) → Output layer
                - **Activation**: ReLU in hidden layers, Softmax in output layer
                - **Regularization**: Dropout layers (0.2) to prevent overfitting
                
                ### Strengths
                - Simple and computationally efficient
                - Works well for tabular data with clear feature relationships
                - Quick to train and deploy
                
                ### Limitations
                - Limited ability to capture complex spatial or temporal patterns
                - May require more feature engineering
                - Often requires more parameters for the same performance as specialized architectures
            """,
            
            'ImprovedFNN': """
                ## Improved Feedforward Neural Network
                
                ### Architecture Overview
                An enhanced version of the standard FNN with deeper architecture and improved regularization.
                
                ### Key Features
                - **Structure**: Input layer → Hidden layer 1 (128 neurons) → Hidden layer 2 (64 neurons) → Hidden layer 3 (32 neurons) → Output layer
                - **Activation**: ReLU in hidden layers, Softmax in output layer
                - **Regularization**: Stronger dropout layers (0.3) to prevent overfitting
                
                ### Strengths
                - Greater capacity to learn complex patterns
                - Better regularization prevents overfitting on smaller datasets
                - Additional layer allows hierarchical feature extraction
                
                ### Limitations
                - More parameters to train compared to standard FNN
                - Longer training time
                - May be prone to vanishing gradients in deeper configurations
            """,
            
            'CNN': """
                ## Convolutional Neural Network (CNN)
                
                ### Architecture Overview
                A neural network using convolutional filters to extract spatial features from input data.
                
                ### Key Features
                - **Structure**: Conv1D layer (64 filters) → MaxPooling → Flatten → Dense layer (32) → Output layer
                - **Filter Size**: 3 units (kernel_size=3)
                - **Pooling**: Max pooling with stride 2 to reduce dimensionality
                
                ### Strengths
                - Effective at capturing local patterns and spatial relationships
                - Parameter sharing reduces model size
                - Translation invariant (can detect patterns regardless of position)
                
                ### Limitations
                - Less effective for purely numerical data without spatial relationships
                - Simpler architecture may miss complex temporal dynamics
            """,
            
            'AdvancedCNN': """
                ## Advanced Convolutional Neural Network
                
                ### Architecture Overview
                A more sophisticated CNN with multiple convolutional layers and global pooling.
                
                ### Key Features
                - **Structure**: Conv1D layer (128 filters) → MaxPooling → Conv1D layer (64 filters) → GlobalAveragePooling → Output layer
                - **Filter Sizes**: 3 units with padding='same' for feature preservation
                - **Pooling**: Global average pooling reduces parameters while maintaining feature importance
                
                ### Strengths
                - Deeper architecture can capture more complex patterns
                - Global average pooling reduces overfitting
                - Padding preserves spatial dimensions better
                
                ### Limitations
                - More computationally intensive
                - May be more than needed for simpler classification tasks
                - Requires more data to generalize effectively
            """,
            
            'LSTM': """
                ## Long Short-Term Memory Network (LSTM)
                
                ### Architecture Overview
                A recurrent neural network architecture designed to process sequential data with the ability to remember important information over long periods.
                
                ### Key Features
                - **Structure**: LSTM layer (64 units) → Dropout (0.2) → Dense layer (32) → Output layer
                - **Memory Cells**: Contains specialized gates (input, forget, output) to control information flow
                - **Sequential Processing**: Processes input features as a sequence
                
                ### Strengths
                - Excellent at capturing temporal dependencies
                - Memory gates help with long-range dependencies
                - Prevents vanishing gradient problem better than standard RNNs
                
                ### Limitations
                - More computationally expensive than feedforward networks
                - More complex to tune and optimize
                - May be unnecessary if temporal relationships aren't important
            """,
            
            'GRU': """
                ## Gated Recurrent Unit Network (GRU)
                
                ### Architecture Overview
                A streamlined version of LSTM with fewer parameters but similar effectiveness for many tasks.
                
                ### Key Features
                - **Structure**: GRU layer (64 units) → Dropout (0.2) → Dense layer (32) → Output layer
                - **Gating Mechanism**: Uses update and reset gates (simpler than LSTM)
                - **Sequential Processing**: Processes input features as a sequence
                
                ### Strengths
                - Fewer parameters than LSTM (more efficient)
                - Still effective at capturing sequential patterns
                - Often trains faster than LSTM
                
                ### Limitations
                - May be less effective than LSTM for very long sequences
                - Still more computationally expensive than non-recurrent models
                - Benefits may be minimal if data has no sequential nature
            """
        }
        
    def create_output_dirs(self):
        """Create the necessary output directories for results"""
        # Main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Subdirectories
        os.makedirs(os.path.join(self.output_dir, 'model_visualizations'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'model_reports'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'performance_plots'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'comparison_reports'), exist_ok=True)
        
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
        
        # Store original dataframes for future reference
        self.train_df = train
        self.valid_df = valid
        self.test_df = test
        
        # Store original class distribution for analysis
        self.class_distribution = df['fault_class'].value_counts().sort_index()
        print(f"Class distribution: {self.class_distribution.to_dict()}")
        
        # Save dataset statistics
        self._save_dataset_info()
        
        return self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.X_test, self.Y_test
    
    def _save_dataset_info(self):
        """Save detailed information about the dataset"""
        dataset_info = {
            "total_samples": len(self.train_df) + len(self.valid_df) + len(self.test_df),
            "train_samples": len(self.train_df),
            "validation_samples": len(self.valid_df),
            "test_samples": len(self.test_df),
            "features": list(self.train_df[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']].columns),
            "class_distribution": self.class_distribution.to_dict(),
            "classes": self.fault_names
        }
        
        # Save as JSON
        with open(os.path.join(self.output_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=4)
        
        # Also save as readable text
        with open(os.path.join(self.output_dir, 'dataset_info.txt'), 'w') as f:
            f.write("FAULT CLASSIFICATION DATASET SUMMARY\n")
            f.write("==================================\n\n")
            f.write(f"Total samples: {dataset_info['total_samples']}\n")
            f.write(f"Training samples: {dataset_info['train_samples']} ({dataset_info['train_samples']/dataset_info['total_samples']*100:.1f}%)\n")
            f.write(f"Validation samples: {dataset_info['validation_samples']} ({dataset_info['validation_samples']/dataset_info['total_samples']*100:.1f}%)\n")
            f.write(f"Test samples: {dataset_info['test_samples']} ({dataset_info['test_samples']/dataset_info['total_samples']*100:.1f}%)\n\n")
            
            f.write("Features used:\n")
            for feature in dataset_info['features']:
                f.write(f"- {feature}\n")
            
            f.write("\nClass distribution:\n")
            for class_id, count in dataset_info['class_distribution'].items():
                f.write(f"- {self.fault_names[int(class_id)]}: {count} samples\n")
                
            # Add feature statistics
            f.write("\nFeature Statistics:\n")
            f.write("------------------\n")
            stats = self.train_df[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']].describe()
            for feature in dataset_info['features']:
                f.write(f"\n{feature}:\n")
                f.write(f"  Mean: {stats.loc['mean', feature]:.4f}\n")
                f.write(f"  Std: {stats.loc['std', feature]:.4f}\n")
                f.write(f"  Min: {stats.loc['min', feature]:.4f}\n")
                f.write(f"  25%: {stats.loc['25%', feature]:.4f}\n")
                f.write(f"  50%: {stats.loc['50%', feature]:.4f}\n")
                f.write(f"  75%: {stats.loc['75%', feature]:.4f}\n")
                f.write(f"  Max: {stats.loc['max', feature]:.4f}\n")
    
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
        if model_type in ['CNN', 'AdvancedCNN', 'LSTM', 'GRU']:
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
        
        # Store model parameters
        self.model_params[model_type] = kwargs.copy()
        
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
        
        # Count layers by type
        layer_counts = {}
        for layer in model.layers:
            layer_type = layer.__class__.__name__
            if layer_type in layer_counts:
                layer_counts[layer_type] += 1
            else:
                layer_counts[layer_type] = 1
        
        self.model_layer_counts[model_type] = layer_counts
        
        # Calculate model size
        self.model_memory_usage[model_type] = model.count_params() * 4 / 1024  # Size in KB (assuming 4 bytes per parameter)
        
        # Store the model
        self.models[model_type] = model
        
        # Create visualizations for the model
        self._visualize_model(model_type, model)
        
        return model
    
    def _visualize_model(self, model_type, model):
        """
        Create various visualizations for the model architecture.
        
        Args:
            model_type (str): Type of model
            model (keras.Model): The model to visualize
        """
        # 1. Standard Keras model plot
        viz_path = os.path.join(self.output_dir, 'model_visualizations', f'{model_type}_architecture.png')
        plot_model(model, to_file=viz_path, show_shapes=True, show_layer_names=True, expand_nested=True, dpi=150)
        
        # 2. Create custom neural network visualization
        self._create_custom_nn_visualization(model_type, model)
        
    def _create_custom_nn_visualization(self, model_type, model):
        """
        Create a custom, more visually appealing neural network visualization.
        
        Args:
            model_type (str): Type of model
            model (keras.Model): The model to visualize
        """
        # Define colors for different layer types
        layer_colors = {
            'Dense': '#3498db',           # Blue
            'Conv1D': '#e74c3c',          # Red
            'LSTM': '#9b59b6',            # Purple
            'GRU': '#8e44ad',             # Dark Purple
            'Dropout': '#95a5a6',         # Gray
            'MaxPooling1D': '#2ecc71',    # Green
            'Flatten': '#f39c12',         # Orange
            'GlobalAveragePooling1D': '#27ae60'  # Dark Green
        }
        
        # Get layer information
        layers = []
        max_neurons = 0
        
        for layer in model.layers:
            layer_type = layer.__class__.__name__
            
            # Determine number of units/neurons for visualization
            if layer_type == 'Dense':
                units = layer.units
            elif layer_type == 'Conv1D':
                units = layer.filters
            elif layer_type in ['LSTM', 'GRU']:
                units = layer.units
            elif layer_type == 'Dropout':
                # Get units from previous layer
                prev_idx = model.layers.index(layer) - 1
                if prev_idx >= 0:
                    prev_layer = model.layers[prev_idx]
                    if hasattr(prev_layer, 'units'):
                        units = prev_layer.units
                    elif hasattr(prev_layer, 'filters'):
                        units = prev_layer.filters
                    else:
                        units = 10  # Default
                else:
                    units = 10
            elif layer_type in ['MaxPooling1D', 'GlobalAveragePooling1D']:
                units = 15  # Arbitrary visualization size
            elif layer_type == 'Flatten':
                units = 20  # Arbitrary visualization size
            else:
                units = 10  # Default for unknown layers
            
            max_neurons = max(max_neurons, units)
            
            layers.append({
                'type': layer_type,
                'units': units,
                'color': layer_colors.get(layer_type, '#34495e')  # Default dark blue
            })
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Constants for drawing
        layer_spacing = 1
        neuron_radius = 0.2
        layer_width = max(3, len(layers))
        max_neurons_to_draw = 15  # Limit neurons for cleaner visualization
        
        # Draw layers
        for i, layer in enumerate(layers):
            units = min(layer['units'], max_neurons_to_draw)
            
            # If too many neurons, add ellipsis
            if layer['units'] > max_neurons_to_draw:
                show_ellipsis = True
                units = max_neurons_to_draw - 3  # Make room for ellipsis
            else:
                show_ellipsis = False
            
            # Calculate positions
            x = i * layer_spacing
            spacing = 8 / max(units, 1)
            
            # Draw neurons
            for j in range(units):
                y = (j - units/2) * spacing
                circle = plt.Circle((x, y), neuron_radius, color=layer['color'], alpha=0.8)
                ax.add_patch(circle)
                
                # Draw connections to previous layer if not first layer
                if i > 0:
                    prev_layer = layers[i-1]
                    prev_units = min(prev_layer['units'], max_neurons_to_draw)
                    
                    # Adjust for ellipsis in previous layer
                    if prev_layer['units'] > max_neurons_to_draw:
                        prev_units = max_neurons_to_draw - 3
                    
                    # Draw connections to previous layer's neurons
                    for k in range(prev_units):
                        prev_y = (k - prev_units/2) * (8 / max(prev_units, 1))
                        line = plt.Line2D([x-layer_spacing, x], [prev_y, y], color='gray', alpha=0.3)
                        ax.add_line(line)
            
            # Add ellipsis if needed
            if show_ellipsis:
                for j in range(3):
                    ellipsis_y = ((units + j) - units/2) * spacing
                    circle = plt.Circle((x, ellipsis_y), neuron_radius * 0.7, color=layer['color'], alpha=0.5)
                    ax.add_patch(circle)
            
            # Label the layer
            plt.text(x, 4.5, layer['type'], ha='center', va='center', 
                    bbox=dict(facecolor=layer['color'], alpha=0.2, boxstyle='round,pad=0.5'))
            
            # For specific layers, add more details
            if layer['type'] in ['Dense', 'Conv1D', 'LSTM', 'GRU']:
                plt.text(x, -4.5, f"{layer['units']} units", ha='center', va='center', 
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
        
        # Set limits and remove axes
        ax.set_xlim(-0.5, (len(layers) - 0.5) * layer_spacing)
        ax.set_ylim(-5, 5)
        ax.axis('off')
        
        # Add title
        plt.title(f"{model_type} Neural Network Architecture", fontsize=16, pad=20)
        
        # Add a legend for layer types
        legend_elements = []
        for layer_type, color in layer_colors.items():
            if any(layer['type'] == layer_type for layer in layers):
                legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=10, label=layer_type))
        
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                 fancybox=True, shadow=True, ncol=min(5, len(legend_elements)))
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_visualizations', f'{model_type}_custom_visualization.png'), dpi=150)
        plt.close()
    
    def train_model(self, model_type, epochs=50, batch_size=64, use_callbacks=True):
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
        
        # Measure prediction speed
        start_time = time.time()
        model.predict(X_valid[:100])
        pred_time = time.time() - start_time
        self.prediction_speeds[model_type] = pred_time / 100  # Average time per sample
        
        # Generate and save training report
        self._generate_training_report(model_type)
        
        return history
    
    def _generate_training_report(self, model_type):
        """
        Generate a detailed training report for the model.
        
        Args:
            model_type (str): Type of model
        """
        # Get training history
        history = self.histories[model_type]
        
        # Create report
        report_path = os.path.join(self.output_dir, 'model_reports', f'{model_type}_training_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Title and basic info
            f.write(f"# {model_type} Model Training Report\n\n")
            f.write(f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n")
            
            # Model description
            f.write("## Model Description\n\n")
            f.write(f"{self.model_descriptions[model_type]}\n\n")
            
            # Extended description
            f.write(self.extended_descriptions[model_type])
            f.write("\n\n")
            
            # Model architecture
            f.write("## Model Architecture\n\n")
            f.write("```\n")
            self.models[model_type].summary(print_fn=lambda x: f.write(x + '\n'))
            f.write("```\n\n")
            
            # Layer counts
            f.write("### Layer Distribution\n\n")
            for layer_type, count in self.model_layer_counts[model_type].items():
                f.write(f"- {layer_type}: {count}\n")
            f.write("\n")
            
            # Model complexity
            f.write("## Model Complexity\n\n")
            f.write(f"- **Total Parameters**: {self.models[model_type].count_params():,}\n")
            f.write(f"- **Trainable Parameters**: {sum(np.prod(v.shape) for v in self.models[model_type].trainable_weights):,}\n")
            f.write(f"- **Non-trainable Parameters**: {sum(np.prod(v.shape) for v in self.models[model_type].non_trainable_weights):,}\n")
            f.write(f"- **Memory Usage**: {self.model_memory_usage[model_type]:.2f} KB\n\n")
            
            # Training details
            f.write("## Training Details\n\n")
            f.write(f"- **Training Time**: {self.training_times[model_type]:.2f} seconds\n")
            f.write(f"- **Prediction Speed**: {self.prediction_speeds[model_type]*1000:.4f} ms per sample\n")
            f.write(f"- **Epochs Trained**: {len(history.history['loss'])}\n")
            
            # Final metrics
            f.write("\n## Final Training Metrics\n\n")
            f.write(f"- **Training Accuracy**: {history.history['accuracy'][-1]:.4f}\n")
            f.write(f"- **Validation Accuracy**: {history.history['val_accuracy'][-1]:.4f}\n")
            f.write(f"- **Training Loss**: {history.history['loss'][-1]:.4f}\n")
            f.write(f"- **Validation Loss**: {history.history['val_loss'][-1]:.4f}\n\n")
            
            # Performance curve descriptions
            f.write("## Learning Curves\n\n")
            
            # Analyze training curves
            max_val_acc = max(history.history['val_accuracy'])
            max_val_acc_epoch = history.history['val_accuracy'].index(max_val_acc) + 1
            
            train_acc_at_max = history.history['accuracy'][max_val_acc_epoch - 1]
            gap = train_acc_at_max - max_val_acc
            
            # Assess overfitting/underfitting
            if gap > 0.1:
                fit_assessment = "The model shows signs of overfitting, as the training accuracy exceeds validation accuracy by a significant margin."
            elif max_val_acc < 0.7:
                fit_assessment = "The model may be underfitting the data, as both training and validation accuracy are relatively low."
            else:
                fit_assessment = "The model shows good fit, with high validation accuracy and minimal gap between training and validation performance."
            
            f.write(f"The model achieved its highest validation accuracy of {max_val_acc:.4f} at epoch {max_val_acc_epoch}. {fit_assessment}\n\n")
            
            # Final notes based on model type
            f.write("## Architecture-Specific Notes\n\n")
            
            if model_type == 'FNN':
                f.write("This standard feedforward neural network provides a good baseline for fault classification. Consider this architecture when computational resources are limited or when a simple, interpretable model is desired.\n\n")
            elif model_type == 'ImprovedFNN':
                f.write("The improved FNN with additional layers and neurons can capture more complex patterns than the standard FNN. This architecture represents a good balance between model complexity and performance.\n\n")
            elif model_type == 'CNN':
                f.write("The convolutional architecture extracts spatial features from the input signals, which can be beneficial for detecting fault patterns that manifest as specific signal shapes or transitions.\n\n")
            elif model_type == 'AdvancedCNN':
                f.write("The advanced CNN with multiple convolutional layers can detect hierarchical features in fault signals. The global average pooling helps make the model more robust to variations in the fault patterns.\n\n")
            elif model_type == 'LSTM':
                f.write("The LSTM network treats the input features as a sequence, potentially capturing temporal dependencies in the fault signals. This can be particularly useful if the order of features is meaningful.\n\n")
            elif model_type == 'GRU':
                f.write("The GRU network provides similar sequential processing capabilities to LSTM but with fewer parameters, potentially offering faster training and inference times while maintaining good performance.\n\n")
            
            f.write("## Recommendations\n\n")
            
            if max_val_acc < 0.8:
                f.write("- Consider increasing model complexity by adding more layers or neurons\n")
                f.write("- Explore different learning rates or optimization algorithms\n")
                f.write("- Review feature scaling and preprocessing steps\n")
            elif gap > 0.1:
                f.write("- Implement stronger regularization to reduce overfitting\n")
                f.write("- Consider reducing model complexity\n")
                f.write("- Explore data augmentation to improve generalization\n")
            else:
                f.write("- The model is performing well; consider this architecture for deployment\n")
                f.write("- Fine-tune hyperparameters for potential incremental improvements\n")
                f.write("- Ensure model is robust to different fault conditions\n")
    
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
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for class_id in range(len(self.fault_names)):
            if str(class_id) in report:
                per_class_metrics[class_id] = {
                    'precision': report[str(class_id)]['precision'],
                    'recall': report[str(class_id)]['recall'],
                    'f1-score': report[str(class_id)]['f1-score'],
                    'support': report[str(class_id)]['support']
                }
        
        # Store results
        self.evaluation_results[model_type] = {
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'accuracy': report['accuracy'],
            'macro_avg_f1': report['macro avg']['f1-score'],
            'weighted_avg_f1': report['weighted avg']['f1-score'],
            'per_class_metrics': per_class_metrics
        }
        
        print(f"\nEvaluation of {model_type} model:")
        print(classification_report(y_test_classes, y_pred_classes))
        print("Confusion Matrix:")
        print(conf_matrix)
        
        # Generate and save evaluation report
        self._generate_evaluation_report(model_type)
        
        # Generate confusion matrix visualization
        self.plot_confusion_matrix(model_type)
        
        # Generate per-class performance visualization
        self._plot_per_class_performance(model_type)
        
        return self.evaluation_results[model_type]
    
    def _generate_evaluation_report(self, model_type):
        """
        Generate a detailed evaluation report for the model.
        
        Args:
            model_type (str): Type of model
        """
        results = self.evaluation_results[model_type]
        
        # Create report
        report_path = os.path.join(self.output_dir, 'model_reports', f'{model_type}_evaluation_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Title and basic info
            f.write(f"# {model_type} Model Evaluation Report\n\n")
            f.write(f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n")
            
            # Overall performance
            f.write("## Overall Performance\n\n")
            f.write(f"- **Accuracy**: {results['accuracy']:.4f}\n")
            f.write(f"- **Macro Average F1**: {results['macro_avg_f1']:.4f}\n")
            f.write(f"- **Weighted Average F1**: {results['weighted_avg_f1']:.4f}\n\n")
            
            # Per-class performance
            f.write("## Per-Class Performance\n\n")
            
            # Create markdown table
            f.write("| Fault Type | Precision | Recall | F1-Score | Support |\n")
            f.write("|------------|-----------|--------|----------|--------|\n")
            
            for class_id, metrics in results['per_class_metrics'].items():
                fault_name = self.fault_names[class_id]
                f.write(f"| {fault_name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1-score']:.4f} | {int(metrics['support'])} |\n")
            
            # Add averages
            f.write(f"| **Macro Avg** | {results['classification_report']['macro avg']['precision']:.4f} | {results['classification_report']['macro avg']['recall']:.4f} | {results['classification_report']['macro avg']['f1-score']:.4f} | {int(results['classification_report']['macro avg']['support'])} |\n")
            f.write(f"| **Weighted Avg** | {results['classification_report']['weighted avg']['precision']:.4f} | {results['classification_report']['weighted avg']['recall']:.4f} | {results['classification_report']['weighted avg']['f1-score']:.4f} | {int(results['classification_report']['weighted avg']['support'])} |\n\n")
            
            # Detailed analysis
            f.write("## Performance Analysis\n\n")
            
            # Find best and worst performing classes
            best_class = max(results['per_class_metrics'].items(), key=lambda x: x[1]['f1-score'])
            worst_class = min(results['per_class_metrics'].items(), key=lambda x: x[1]['f1-score'])
            
            f.write(f"- Best performing class: **{self.fault_names[best_class[0]]}** with F1-score of {best_class[1]['f1-score']:.4f}\n")
            f.write(f"- Worst performing class: **{self.fault_names[worst_class[0]]}** with F1-score of {worst_class[1]['f1-score']:.4f}\n\n")
            
            # Analyze confusion patterns
            conf_matrix = results['confusion_matrix']
            most_confused = (-1, -1, -1)  # (true_class, pred_class, count)
            
            for i in range(len(conf_matrix)):
                for j in range(len(conf_matrix)):
                    if i != j and conf_matrix[i, j] > most_confused[2]:
                        most_confused = (i, j, conf_matrix[i, j])
            
            if most_confused[2] > 0:
                f.write(f"- Most common confusion: {most_confused[2]} instances of **{self.fault_names[most_confused[0]]}** were misclassified as **{self.fault_names[most_confused[1]]}**\n\n")
            
            # Recommendations based on results
            f.write("## Recommendations\n\n")
            
            if results['accuracy'] < 0.9:
                f.write("- Consider model improvements to increase overall accuracy\n")
            
            if worst_class[1]['f1-score'] < 0.8:
                f.write(f"- Focus on improving performance for {self.fault_names[worst_class[0]]} fault type\n")
                f.write(f"- Consider collecting more training examples for {self.fault_names[worst_class[0]]}\n")
            
            if most_confused[2] > 5:
                f.write(f"- Investigate why {self.fault_names[most_confused[0]]} is being confused with {self.fault_names[most_confused[1]]}\n")
                f.write("- Consider adding engineered features to better distinguish these classes\n")
    
    def _plot_per_class_performance(self, model_type):
        """
        Create a visualization of per-class performance metrics.
        
        Args:
            model_type (str): Type of model
        """
        results = self.evaluation_results[model_type]
        
        # Extract per-class metrics
        classes = []
        precision = []
        recall = []
        f1_score = []
        
        for class_id, metrics in results['per_class_metrics'].items():
            classes.append(self.fault_names[class_id])
            precision.append(metrics['precision'])
            recall.append(metrics['recall'])
            f1_score.append(metrics['f1-score'])
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(classes))
        width = 0.25
        
        plt.bar(x - width, precision, width, label='Precision', color='#3498db')
        plt.bar(x, recall, width, label='Recall', color='#2ecc71')
        plt.bar(x + width, f1_score, width, label='F1-Score', color='#e74c3c')
        
        plt.xlabel('Fault Type')
        plt.ylabel('Score')
        plt.title(f'{model_type} - Per-Class Performance Metrics')
        plt.xticks(x, classes, rotation=45)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_plots', f'{model_type}_per_class_performance.png'))
        plt.close()
    
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
                'Training Time (s)': self.training_times.get(model_type, float('nan')),
                'Prediction Speed (ms)': self.prediction_speeds.get(model_type, float('nan')) * 1000,
                'Parameter Count': self.models[model_type].count_params(),
                'Memory (KB)': self.model_memory_usage.get(model_type, float('nan'))
            })
            
        comparison_df = pd.DataFrame(comparisons)
        
        # Sort by weighted F1 score
        comparison_df = comparison_df.sort_values('Weighted Avg F1', ascending=False)
        
        # Find best model
        self.best_model_name = comparison_df.iloc[0]['Model']
        
        # Generate comparison reports and visualizations
        self._generate_comparison_report(comparison_df)
        self._plot_model_comparison(comparison_df)
        self._create_radar_chart_comparison(comparison_df)
        self._plot_parameter_distribution(comparison_df)  # New visualization
        
        return comparison_df
    
    def _generate_comparison_report(self, comparison_df):
        """
        Generate a detailed report comparing all models.
        
        Args:
            comparison_df (pd.DataFrame): Comparison dataframe
        """
        report_path = os.path.join(self.output_dir, 'comparison_reports', 'model_comparison_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Title and basic info
            f.write("# Model Comparison Report\n\n")
            f.write(f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n")
            
            # Overall rankings
            f.write("## Model Rankings (by Weighted F1-Score)\n\n")
            
            # Convert DataFrame to markdown table
            f.write("| Rank | Model | Accuracy | Macro Avg F1 | Weighted Avg F1 | Training Time (s) | Prediction Speed (ms) | Parameters | Memory (KB) |\n")
            f.write("|------|-------|----------|--------------|-----------------|-------------------|----------------------|------------|------------|\n")
            
            for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
                f.write(f"| {i} | {row['Model']} | {row['Accuracy']:.4f} | {row['Macro Avg F1']:.4f} | {row['Weighted Avg F1']:.4f} | {row['Training Time (s)']:.2f} | {row['Prediction Speed (ms)']:.4f} | {int(row['Parameter Count']):,} | {row['Memory (KB)']:.2f} |\n")
            
            f.write("\n")
            
            # Best model in each category
            f.write("## Best Models by Category\n\n")
            
            categories = {
                'Accuracy': 'highest accuracy score',
                'Weighted Avg F1': 'highest weighted F1-score',
                'Macro Avg F1': 'highest macro-average F1-score',
                'Training Time (s)': 'fastest training time',
                'Prediction Speed (ms)': 'fastest prediction speed',
                'Parameter Count': 'most efficient (fewest parameters)',
                'Memory (KB)': 'smallest memory footprint'
            }
            
            for metric, description in categories.items():
                if metric in ['Training Time (s)', 'Prediction Speed (ms)', 'Parameter Count', 'Memory (KB)']:
                    best_model = comparison_df.loc[comparison_df[metric].idxmin()]['Model']
                    best_value = comparison_df.loc[comparison_df[metric].idxmin()][metric]
                else:
                    best_model = comparison_df.loc[comparison_df[metric].idxmax()]['Model']
                    best_value = comparison_df.loc[comparison_df[metric].idxmax()][metric]
                
                if metric in ['Parameter Count']:
                    f.write(f"- **{description}**: {best_model} ({int(best_value):,} parameters)\n")
                elif metric in ['Training Time (s)', 'Memory (KB)']:
                    f.write(f"- **{description}**: {best_model} ({best_value:.2f} {metric[-4:]})\n")
                elif metric in ['Prediction Speed (ms)']:
                    f.write(f"- **{description}**: {best_model} ({best_value:.4f} ms per sample)\n")
                else:
                    f.write(f"- **{description}**: {best_model} ({best_value:.4f})\n")
            
            f.write("\n")
            
            # Creative analogies
            f.write("## Creative Model Analogies\n\n")
            analogies = {
                'FNN': "🚗 Reliable Sedan - Gets you there efficiently but without flair",
                'ImprovedFNN': "🚙 SUV - More powerful version of the sedan",
                'CNN': "🛩️ Aircraft - Great for pattern recognition in spatial data",
                'AdvancedCNN': "🚀 Spaceship - Advanced version for complex patterns",
                'LSTM': "🕰️ Antique Clock - Excellent for temporal patterns but complex",
                'GRU': "⏱️ Modern Watch - Efficient time-aware processing"
            }
            
            for model, analogy in analogies.items():
                f.write(f"- **{model}**: {analogy}\n")
            f.write("\n")
            
            # Model architectures comparison
            f.write("## Architecture Comparison\n\n")
            
            for model_type in self.models.keys():
                f.write(f"### {model_type}\n\n")
                f.write(f"{self.model_descriptions[model_type]}\n\n")
                
                # Add layer counts
                f.write("#### Layer Composition\n\n")
                for layer_type, count in self.model_layer_counts[model_type].items():
                    f.write(f"- {layer_type}: {count}\n")
                
                f.write("\n")
            
            # Trade-off analysis
            f.write("## Performance Trade-offs\n\n")
            
            # Accuracy vs Speed
            fastest_model = comparison_df.loc[comparison_df['Prediction Speed (ms)'].idxmin()]['Model']
            most_accurate = comparison_df.loc[comparison_df['Accuracy'].idxmax()]['Model']
            
            f.write("### Accuracy vs. Speed\n\n")
            if fastest_model == most_accurate:
                f.write(f"The {fastest_model} model offers the best of both worlds with highest accuracy and fastest prediction speed.\n\n")
            else:
                fastest_accuracy = comparison_df.loc[comparison_df['Prediction Speed (ms)'].idxmin()]['Accuracy']
                most_accurate_speed = comparison_df.loc[comparison_df['Accuracy'].idxmax()]['Prediction Speed (ms)']
                
                accuracy_diff = comparison_df.loc[comparison_df['Accuracy'].idxmax()]['Accuracy'] - fastest_accuracy
                speed_factor = most_accurate_speed / comparison_df.loc[comparison_df['Prediction Speed (ms)'].idxmin()]['Prediction Speed (ms)']
                
                f.write(f"- The {fastest_model} model is the fastest (prediction time: {comparison_df.loc[comparison_df['Prediction Speed (ms)'].idxmin()]['Prediction Speed (ms)']:.4f} ms) but sacrifices {accuracy_diff*100:.2f}% in accuracy compared to the most accurate model.\n")
                f.write(f"- The {most_accurate} model achieves the highest accuracy ({comparison_df.loc[comparison_df['Accuracy'].idxmax()]['Accuracy']:.4f}) but is {speed_factor:.1f}x slower than the fastest model.\n\n")
            
            # Complexity vs Performance
            smallest_model = comparison_df.loc[comparison_df['Parameter Count'].idxmin()]['Model']
            largest_model = comparison_df.loc[comparison_df['Parameter Count'].idxmax()]['Model']
            
            f.write("### Complexity vs. Performance\n\n")
            smallest_f1 = comparison_df.loc[comparison_df['Parameter Count'].idxmin()]['Weighted Avg F1']
            largest_f1 = comparison_df.loc[comparison_df['Parameter Count'].idxmax()]['Weighted Avg F1']
            
            param_ratio = comparison_df.loc[comparison_df['Parameter Count'].idxmax()]['Parameter Count'] / comparison_df.loc[comparison_df['Parameter Count'].idxmin()]['Parameter Count']
            
            if largest_f1 > smallest_f1:
                f1_improvement = (largest_f1 - smallest_f1) / smallest_f1 * 100
                f.write(f"- The {largest_model} model has {param_ratio:.1f}x more parameters than the {smallest_model} model, resulting in a {f1_improvement:.2f}% improvement in F1-score.\n")
                f.write(f"- This suggests that the added complexity is {f1_improvement/param_ratio:.2f}% efficient in terms of F1-score improvement per parameter ratio increase.\n\n")
            else:
                f1_decline = (smallest_f1 - largest_f1) / largest_f1 * 100
                f.write(f"- Despite having {param_ratio:.1f}x more parameters, the {largest_model} model performs {f1_decline:.2f}% worse than the simpler {smallest_model} model.\n")
                f.write("- This suggests that simply increasing model complexity does not guarantee better performance for this classification task.\n\n")
            
            # Overall recommendations
            f.write("## Overall Recommendations\n\n")
            
            # Best overall model
            f.write(f"### Best Overall Model: {self.best_model_name}\n\n")
            f.write(f"The {self.best_model_name} model provides the best balance of accuracy, F1-score, and efficiency for fault classification.\n\n")
            
            # Use case specific recommendations
            f.write("### Use Case Specific Recommendations\n\n")
            f.write("- **For resource-constrained environments**: ")
            f.write(f"{comparison_df.loc[comparison_df['Memory (KB)'].idxmin()]['Model']} offers the smallest footprint.\n")
            f.write("- **For real-time applications**: ")
            f.write(f"{comparison_df.loc[comparison_df['Prediction Speed (ms)'].idxmin()]['Model']} provides the fastest inference time.\n")
            f.write("- **For maximum accuracy**: ")
            f.write(f"{comparison_df.loc[comparison_df['Accuracy'].idxmax()]['Model']} achieves the highest classification accuracy.\n")
            f.write("- **For balanced performance**: ")
            f.write(f"{comparison_df.loc[comparison_df['Weighted Avg F1'].idxmax()]['Model']} offers the best weighted F1-score across all fault types.\n\n")
    
    def _plot_model_comparison(self, comparison_df):
        """
        Create enhanced visual comparisons between models.
        
        Args:
            comparison_df (pd.DataFrame): Comparison dataframe
        """
        # 1. Performance metrics comparison
        plt.figure(figsize=(14, 8))
        
        # Create subplots for different metrics
        metrics = ['Accuracy', 'Macro Avg F1', 'Weighted Avg F1']
        
        for i, metric in enumerate(metrics):
            plt.subplot(1, 3, i+1)
            
            # Create bar chart with custom colors
            colors = plt.cm.viridis(np.linspace(0, 0.8, len(comparison_df)))
            bars = plt.bar(comparison_df['Model'], comparison_df[metric], color=colors)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
            
            plt.title(f'Model Comparison - {metric}')
            plt.xticks(rotation=45)
            plt.ylim(0, max(comparison_df[metric]) * 1.1)  # Add some padding
            plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comparison_reports', 'performance_comparison.png'))
        plt.close()
        
        # 2. Efficiency metrics comparison
        plt.figure(figsize=(14, 8))
        
        # Create subplots for different metrics
        eff_metrics = ['Training Time (s)', 'Prediction Speed (ms)', 'Memory (KB)']
        
        for i, metric in enumerate(eff_metrics):
            plt.subplot(1, 3, i+1)
            
            # Sort by this metric
            df_sorted = comparison_df.sort_values(metric)
            
            # Create bar chart with custom colors - use reversed colormap for efficiency metrics
            colors = plt.cm.plasma(np.linspace(0.8, 0, len(df_sorted)))
            bars = plt.bar(df_sorted['Model'], df_sorted[metric], color=colors)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if metric == 'Prediction Speed (ms)':
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                            f'{height:.4f}', ha='center', va='bottom', fontsize=9)
                else:
                    plt.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            
            plt.title(f'Model Comparison - {metric}')
            plt.xticks(rotation=45)
            plt.ylim(0, max(df_sorted[metric]) * 1.1)  # Add some padding
            plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comparison_reports', 'efficiency_comparison.png'))
        plt.close()
        
        # 3. Combined performance and parameter count visualization
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot of Weighted F1 vs Parameters with Prediction Speed as color
        scatter = plt.scatter(
            comparison_df['Parameter Count'], 
            comparison_df['Weighted Avg F1'],
            c=comparison_df['Prediction Speed (ms)'],
            s=comparison_df['Accuracy'] * 500,  # Size represents accuracy
            alpha=0.7,
            cmap='viridis'
        )
        
        # Add labels and titles
        plt.xlabel('Parameter Count')
        plt.ylabel('Weighted Avg F1 Score')
        plt.title('Model Performance vs Complexity')
        plt.colorbar(scatter, label='Prediction Speed (ms)')
        
        # Add model name annotations
        for i, row in comparison_df.iterrows():
            plt.annotate(row['Model'], 
                        (row['Parameter Count'], row['Weighted Avg F1']),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=9)
        
        # Create legend for accuracy (size)
        acc_min = comparison_df['Accuracy'].min()
        acc_max = comparison_df['Accuracy'].max()
        
        # Create dummy scatter plots for legend
        for acc in [acc_min, (acc_min+acc_max)/2, acc_max]:
            plt.scatter([], [], c='k', alpha=0.5, s=acc*500,
                       label=f'{acc:.2f}')
        
        plt.legend(title='Accuracy', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comparison_reports', 'performance_vs_complexity.png'))
        plt.close()
    
    def _create_radar_chart_comparison(self, comparison_df):
        """
        Create a radar chart comparing multiple model metrics.
        
        Args:
            comparison_df (pd.DataFrame): Comparison dataframe
        """
        # Select metrics to compare
        metrics = ['Accuracy', 'Macro Avg F1', 'Weighted Avg F1', 
                   'Prediction Speed (ms)', 'Training Time (s)']
        
        # Normalize metrics (higher is better for all except speed/time)
        df_normalized = comparison_df.copy()
        
        for metric in metrics:
            if metric in ['Prediction Speed (ms)', 'Training Time (s)']:
                # For speed/time, lower is better
                df_normalized[metric] = 1 - (df_normalized[metric] - df_normalized[metric].min()) / \
                                       (df_normalized[metric].max() - df_normalized[metric].min())
            else:
                # For performance metrics, higher is better
                df_normalized[metric] = (df_normalized[metric] - df_normalized[metric].min()) / \
                                      (df_normalized[metric].max() - df_normalized[metric].min())
        
        # Number of variables
        categories = metrics
        N = len(categories)
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Calculate angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the loop
        
        # Plot each model
        colors = plt.cm.viridis(np.linspace(0, 1, len(comparison_df)))
        
        for i, (_, row) in enumerate(comparison_df.iterrows()):
            values = df_normalized.loc[row.name, metrics].values.flatten().tolist()
            values += values[:1]  # Complete the loop
            
            ax.plot(angles, values, color=colors[i], linewidth=2, linestyle='solid', 
                   label=row['Model'])
            ax.fill(angles, values, color=colors[i], alpha=0.25)
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Add title
        plt.title('Model Comparison Radar Chart', size=20, y=1.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comparison_reports', 'radar_chart_comparison.png'))
        plt.close()
    
    def _plot_parameter_distribution(self, comparison_df):
        """
        Create a sunburst chart showing parameter distribution across models.
        
        Args:
            comparison_df (pd.DataFrame): Comparison dataframe
        """
        plt.figure(figsize=(12, 8))
        
        # Calculate sizes based on parameter count
        sizes = comparison_df['Parameter Count'].values
        labels = [f"{row['Model']}\n{row['Parameter Count']:,}" for _, row in comparison_df.iterrows()]
        colors = plt.cm.tab20c(np.linspace(0, 1, len(sizes)))
        
        # Create treemap
        squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, 
                     text_kwargs={'fontsize':10, 'wrap':True})
        
        plt.title('Model Parameter Distribution (Treemap)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comparison_reports', 'parameter_treemap.png'))
        plt.close()
    
    def plot_confusion_matrix(self, model_type):
        """
        Plot confusion matrix for a specific model.
        
        Args:
            model_type (str): Type of model
        """
        if model_type not in self.evaluation_results:
            raise ValueError(f"No evaluation results for {model_type}. Call evaluate_model first.")
        
        cm = self.evaluation_results[model_type]['confusion_matrix']
        classes = [self.fault_names[i] for i in range(len(self.fault_names))]
        
        # Create custom colormap
        colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                    xticklabels=classes, yticklabels=classes,
                    cbar_kws={'label': 'Number of Samples'})
        
        plt.title(f'Confusion Matrix - {model_type} Model')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_plots', f'{model_type}_confusion_matrix.png'))
        plt.close()
    
    def plot_training_history(self, model_type):
        """
        Plot training and validation metrics over epochs.
        
        Args:
            model_type (str): Type of model
        """
        if model_type not in self.histories:
            raise ValueError(f"No training history for {model_type}. Call train_model first.")
        
        history = self.histories[model_type]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy', color='#3498db')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#2ecc71')
        ax1.set_title(f'{model_type} - Accuracy Over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss', color='#e74c3c')
        ax2.plot(history.history['val_loss'], label='Validation Loss', color='#9b59b6')
        ax2.set_title(f'{model_type} - Loss Over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_plots', f'{model_type}_training_history.png'))
        plt.close()
    
    def save_model(self, model_type):
        """
        Save a trained model to disk.
        
        Args:
            model_type (str): Type of model to save
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not built yet. Call build_model first.")
        
        model_path = os.path.join(self.output_dir, f'{model_type}_model.h5')
        save_model(self.models[model_type], model_path)
        print(f"Saved {model_type} model to {model_path}")
    
    def load_saved_model(self, model_type):
        """
        Load a previously saved model from disk.
        
        Args:
            model_type (str): Type of model to load
        """
        model_path = os.path.join(self.output_dir, f'{model_type}_model.h5')
        self.models[model_type] = load_model(model_path)
        print(f"Loaded {model_type} model from {model_path}")
    
    def get_best_model(self):
        """
        Get the best performing model based on comparison.
        
        Returns:
            tuple: (model_name, model_object)
        """
        if not self.best_model_name:
            self.compare_models()
        
        return self.best_model_name, self.models[self.best_model_name]
    
    def analyze_unknown_samples(self):
        """
        Analyze samples with unknown fault types using the best model.
        """
        if not self.best_model_name:
            self.compare_models()
        
        if self.df_unknown.empty:
            print("No unknown samples to analyze.")
            return None
        
        # Get best model
        model = self.models[self.best_model_name]
        
        # Prepare unknown samples
        X_unknown = self.df_unknown[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']].values
        X_unknown = StandardScaler().fit_transform(X_unknown)
        X_unknown = self._reshape_for_model(X_unknown, self.best_model_name)
        
        # Make predictions
        y_pred = model.predict(X_unknown)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Add predictions to dataframe
        self.df_unknown['predicted_class'] = y_pred_classes
        self.df_unknown['predicted_fault'] = self.df_unknown['predicted_class'].map(self.fault_names)
        
        # Save results
        output_path = os.path.join(self.output_dir, 'unknown_samples_analysis.csv')
        self.df_unknown.to_csv(output_path, index=False)
        
        print(f"Saved analysis of {len(self.df_unknown)} unknown samples to {output_path}")
        
        return self.df_unknown

def main():
    # Initialize the fault classifier with your dataset path
    data_path = 'classData.csv'  # Replace with your actual data file path
    classifier = EnhancedFaultClassifier(data_path)
    
    # Load and preprocess the data
    classifier.load_data()
    
    # Define the models to train and evaluate
    models_to_train = ['FNN', 'ImprovedFNN', 'CNN', 'AdvancedCNN', 'LSTM', 'GRU']
    
    # Train and evaluate each model
    for model_type in models_to_train:
        print(f"\n{'='*50}")
        print(f"Processing {model_type} model")
        print(f"{'='*50}")
        
        # Build the model
        classifier.build_model(model_type)
        
        # Train the model
        classifier.train_model(model_type, epochs=50, batch_size=64)
        
        # Evaluate the model
        classifier.evaluate_model(model_type)
        
        # Plot training history
        classifier.plot_training_history(model_type)
        
        # Save the model
        classifier.save_model(model_type)
    
    # Compare all models
    comparison_results = classifier.compare_models()
    print("\nModel Comparison Results:")
    print(comparison_results)
    
    # Get the best model
    best_model_name, best_model = classifier.get_best_model()
    print(f"\nBest performing model: {best_model_name}")
    
    # Analyze unknown samples (if any)
    unknown_results = classifier.analyze_unknown_samples()
    if unknown_results is not None:
        print("\nUnknown samples analysis:")
        print(unknown_results[['fault_type', 'predicted_fault']].head())
    
    print("\nAll operations completed successfully!")

if __name__ == '__main__':
    main()