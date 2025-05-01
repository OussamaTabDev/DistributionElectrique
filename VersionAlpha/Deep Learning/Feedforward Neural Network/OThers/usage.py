"""
Example script showing how to use the FaultClassifier class for a specific model.
"""

from fault_classification import FaultClassifier
import matplotlib.pyplot as plt

# Initialize the classifier
classifier = FaultClassifier('classData.csv')

# Load and preprocess the data
classifier.load_data()

# Build and train only a specific model (e.g., ImprovedFNN)
print("\nBuilding and training ImprovedFNN model...")
classifier.build_model('ImprovedFNN')
classifier.train_model('ImprovedFNN', epochs=30)
classifier.evaluate_model('ImprovedFNN')
classifier.plot_training_history('ImprovedFNN')
classifier.plot_confusion_matrix('ImprovedFNN')

# Or alternatively, you can build and train specific models with custom parameters
print("\nBuilding and training custom LSTM model...")
classifier.build_model('LSTM', units_1=128, dropout_1=0.3, learning_rate=0.0005)
classifier.train_model('LSTM', epochs=30, batch_size=64)
classifier.evaluate_model('LSTM')

# Save the best model
best_model_path = classifier.save_best_model()
print(f"Best model saved to: {best_model_path}")

# You can also use the classifier for prediction with new data
# (This is just an example, you'd need to implement the prediction functionality)
print("\nExample of how you would use the model for prediction:")
print("model = classifier.models['ImprovedFNN']")
print("# Reshape and preprocess new data as needed")
print("predictions = model.predict(new_data)")