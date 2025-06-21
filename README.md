# Electrical Distribution Fault Analysis System
![status](https://img.shields.io/badge/status-completed-brightgreen)
## Project Overview
This project implements a comprehensive system for analyzing and classifying faults in electrical distribution networks using various Machine Learning and Deep Learning approaches. The system includes supervised learning, unsupervised learning, and deep learning models, along with a web interface for real-time fault detection and visualization.

## Project Structure
```
├── classData.csv                    # Main dataset file
├── Deep Learning/                   # Deep Learning implementation
│   ├── Autoencoders/               # Autoencoder models
│   └── Feedforward Neural Network/  # FNN implementations and results
│       └── Fault_Classification_Results/
│           ├── Various model files (.h5)
│           └── Comparison reports and visualizations
├── Machine Learning/
│   ├── Suprivised ML/              # Supervised learning implementations
│   │   └── SML/                    # Visualization and results
│   └── Usuprivsed ML/              # Unsupervised learning implementations
│       └── UML/    
├── Rapport + Presentation + Video  # Clustering and anomaly detection results
└── Website/                        # Web interface implementation
```

## Features
- **Multiple Model Implementations:**
  - Supervised Learning Models (Random Forest, SVM, XGBoost, etc.)
  - Unsupervised Learning (K-Means, DBSCAN, GMM)
  - Deep Learning Models (CNN, LSTM, GRU, Autoencoders)
  
- **Comprehensive Analysis:**
  - Fault Classification
  - Anomaly Detection
  - Pattern Recognition
  - Performance Comparisons

- **Visualization:**
  - Confusion Matrices
  - Distribution Plots
  - Cluster Visualizations
  - Performance Metrics

- **Web Interface:**
  - Real-time Fault Detection
  - Interactive Visualizations
  - Model Performance Monitoring

## Models Implemented

### Supervised Learning
- Random Forest
- Support Vector Machines (Linear, RBF)
- XGBoost
- K-Nearest Neighbors
- Decision Trees
- Logistic Regression
- Naive Bayes
- SGD Classifier

### Unsupervised Learning
- K-Means Clustering
- DBSCAN
- Gaussian Mixture Models
- Agglomerative Clustering

### Deep Learning
- Convolutional Neural Networks (CNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Units (GRU)
- Feedforward Neural Networks
- Autoencoders

## Installation

1. Clone the repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Interface
```bash
cd Website
streamlit run app.py
```

### Jupyter Notebooks
The project includes several Jupyter notebooks for different analyses:
- `Deep Learning/Feedforward Neural Network/main.ipynb`
- `Machine Learning/Suprivised ML/main.ipynb`
- `Machine Learning/Usuprivsed ML/main.ipynb`

## Results
The project includes comprehensive visualization and comparison of different models:
- Model comparison plots
- Confusion matrices
- Distribution analyses
- Cluster visualizations
- Performance metrics

## Documentation
Detailed documentation and reports can be found in:
- `Rapport + Presentation + Video/Rapport.pdf`
- Model-specific documentation in respective directories
- Web interface documentation in `Website/README.md`

## Requirements
See `requirements.txt` for a complete list of dependencies.

## Contributing
This project was developed as part of a final year project (PFE) focusing on electrical distribution fault analysis.

## License
[MIT License](LICENSE)

## Contact
For any queries regarding this project, please refer to the documentation or contact the repository maintainer.

---
Note: This project is completed and was developed as part of a Professional Final Year Project (PFE) focusing on electrical distribution network fault analysis and classification.
