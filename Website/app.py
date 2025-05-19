import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from joblib import load as joblib_load
import json
import os

# Configuration
MODEL_DIRS = {
    "Deep Learning": "Deep Learning/Feedforward Neural Network/Fault_Classification_Results/",
    "Machine Learning": "Machine Learning/Suprivised ML/FullRapport/models"
}
MODEL_INFO_FILE = "Website/model_info.json"

# Load model information
@st.cache_data
def load_model_info():
    try:
        with open(MODEL_INFO_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"Model information file not found: {MODEL_INFO_FILE}")
        return {}

# Fault type mapping with detailed descriptions
FAULT_MAP = {
    0: {
        "name": "No Fault",
        "description": "The system is operating normally with balanced currents and voltages.",
        "symptoms": [
            "Balanced three-phase currents",
            "Nominal voltage levels",
            "No protective device operation"
        ],
        "actions": [
            "Continue normal operation",
            "Monitor system parameters"
        ]
    },
    1: {
        "name": "LG (Line to Ground)",
        "description": "A single line-to-ground fault occurs when one conductor comes into contact with ground.",
        "symptoms": [
            "High current in faulted phase",
            "Voltage collapse in faulted phase",
            "Ground fault protection activation"
        ],
        "actions": [
            "Isolate the affected line",
            "Check insulation of faulted phase",
            "Inspect grounding systems"
        ]
    },
    2: {
        "name": "LL (Line to Line AB)",
        "description": "A line-to-line fault occurs when two conductors come into contact with each other.",
        "symptoms": [
            "High currents in both faulted phases",
            "Voltage drop in both phases",
            "Phase protection activation"
        ],
        "actions": [
            "Isolate the affected circuit",
            "Check for physical conductor damage",
            "Inspect insulation between phases"
        ]
    },
    3: {
        "name": "LL (Line to Line BC)",
        "description": "A line-to-line fault occurs when two conductors come into contact with each other.",
        "symptoms": [
            "High currents in both faulted phases",
            "Voltage drop in both phases",
            "Phase protection activation"
        ],
        "actions": [
            "Isolate the affected circuit",
            "Check for physical conductor damage",
            "Inspect insulation between phases"
        ]
    },
    4: {
        "name": "LLG (Line-Line to Ground)",
        "description": "A double line-to-ground fault occurs when two conductors contact both each other and ground.",
        "symptoms": [
            "High currents in both faulted phases",
            "Voltage collapse in faulted phases",
            "Both phase and ground protection activation"
        ],
        "actions": [
            "Immediate isolation required",
            "Check for multiple insulation failures",
            "Inspect nearby lightning arresters"
        ]
    },
    5: {
        "name": "LLL (Three-phase)",
        "description": "A three-phase fault occurs when all three conductors come into contact with each other.",
        "symptoms": [
            "Extremely high currents in all phases",
            "Complete voltage collapse",
            "Instantaneous protection operation"
        ],
        "actions": [
            "Emergency shutdown required",
            "Check for catastrophic insulation failure",
            "Inspect for physical damage to busbars"
        ]
    },
    6: {
        "name": "LLLG (Three-phase to Ground)",
        "description": "A three-phase-to-ground fault is the most severe fault where all phases contact each other and ground.",
        "symptoms": [
            "Maximum possible fault currents",
            "Complete voltage collapse",
            "All protection systems activated"
        ],
        "actions": [
            "Emergency shutdown and isolation",
            "Full system inspection required",
            "Check for explosion or fire damage"
        ]
    }
}

# Available models
@st.cache_data
def get_available_models():
    models = {}
    model_info = load_model_info()
    
    for model_type, model_dir in MODEL_DIRS.items():
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.h5') or file.endswith('.joblib'):
                    model_name = file.split('.')[0]
                    models[model_name] = {
                        "path": os.path.join(model_dir, file),
                        "type": model_type,
                        "info": model_info.get(model_name, {})
                    }
        else:
            st.sidebar.warning(f"Directory not found: {model_dir}")
    
    return models

# Load the trained model
@st.cache_resource
def load_trained_model(model_path):
    try:
        if model_path.endswith('.h5'):
            return load_model(model_path)
        elif model_path.endswith('.joblib'):
            return joblib_load(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path}. Only .h5 and .joblib files are supported.")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# FIXED: Improved scaler to better handle the range of measurements in the dataset
@st.cache_resource
def get_fitted_scaler():
    scaler = StandardScaler()
    # Fit the scaler with a wider range of power system measurements
    typical_measurements = np.array([
        # Normal operating conditions
        [.1, .1, .1, 0.1, 0.1, 0.1],
        # Various fault conditions - expanded ranges
        [-700.0, -150.0, -100.0, 0.3, -0.1, -0.3],  # LG fault (negative currents)
        [700.0, -850.0, -30.0, -0.01, -0.12, 0.13],  # LL fault (mixed currents)
        [-60.0, -140.0, 200.0, -0.5, -0.04, 0.55],   # LLG fault
        [500.0, 500.0, 500.0, 0.01, 0.01, 0.01],     # LLL fault
        [700.0, 700.0, 700.0, 0.0, 0.0, 0.0]         # LLLG fault
    ])
    scaler.fit(typical_measurements)
    return scaler

# FIXED: Improved preprocessing function with better input validation
def preprocess_input(input_data):
    """Preprocess user input for prediction using pre-fitted scaler"""
    # Validate input data
    if len(input_data) != 6:
        st.error("Input data must have exactly 6 features (Ia, Ib, Ic, Va, Vb, Vc)")
        return None
    
    try:
        # Convert all inputs to float to ensure compatibility
        input_data = [float(x) for x in input_data]
        scaler = get_fitted_scaler()
        features = np.array(input_data).reshape(1, -1)
        scaled_features = scaler.transform(features)
        return scaled_features
    except Exception as e:
        st.error(f"Error preprocessing input data: {str(e)}")
        return None

# FIXED: Improved prediction function with better error handling
def predict_fault(model, input_data, validate=False, true_class=None):
    """Make prediction using the model and optionally validate against true class"""
    if model is None:
        st.error("Model not loaded properly!")
        return 0, 0, np.zeros(len(FAULT_MAP)), None
    
    processed_data = preprocess_input(input_data)
    if processed_data is None:
        return 0, 0, np.zeros(len(FAULT_MAP)), None
        
    try:
        prediction = model.predict(processed_data)
        # Handle different prediction formats for different model types
        if len(prediction.shape) > 1 and prediction.shape[1] > 1:
            # For models that return probabilities for each class
            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction))
            probabilities = prediction[0]
        else:
            # For models that return class index directly
            predicted_class = int(np.round(prediction[0][0]) if len(prediction.shape) > 1 else np.round(prediction[0]))
            confidence = 1.0  # We don't have confidence for these models
            # Create a one-hot encoded array for probabilities
            probabilities = np.zeros(len(FAULT_MAP))
            probabilities[predicted_class] = 1.0
        
        # Ensure predicted class is within valid range
        if predicted_class < 0 or predicted_class >= len(FAULT_MAP):
            st.warning(f"Model predicted an invalid class: {predicted_class}. Using 'No Fault' as default.")
            predicted_class = 0
            confidence = 0.5
            probabilities = np.zeros(len(FAULT_MAP))
            probabilities[0] = 1.0
            
        # Validation metrics if true class is provided
        validation_metrics = None
        if validate and true_class is not None:
            validation_metrics = {
                'correct': predicted_class == true_class,
                'true_class': true_class,
                'predicted_class': predicted_class,
                'confidence': confidence
            }
            
        return predicted_class, confidence, probabilities, validation_metrics
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return 0, 0, np.zeros(len(FAULT_MAP)), None

def display_fault_info(fault_class):
    """Display detailed information about the detected fault"""
    # Ensure the fault class exists in our mapping
    if fault_class not in FAULT_MAP:
        st.error(f"Unknown fault class: {fault_class}")
        fault_class = 0  # Default to "No Fault"
        
    fault_info = FAULT_MAP[fault_class]
    
    st.subheader(f"Fault Information: {fault_info['name']}")
    st.markdown(f"**Description:** {fault_info['description']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Typical Symptoms:**")
        for symptom in fault_info['symptoms']:
            st.markdown(f"- {symptom}")
    
    with col2:
        st.markdown("**Recommended Actions:**")
        for action in fault_info['actions']:
            st.markdown(f"- {action}")
    
    if fault_class != 0:  # If not "No Fault"
        st.warning("⚠️ Immediate attention required!")

def display_model_info(model_name, model_info):
    """Display information about the selected model"""
    st.subheader("Model Information")
    
    if model_info:
        st.markdown(f"**Name:** {model_name}")
        st.markdown(f"**Type:** {model_info.get('type', 'N/A')}")
        st.markdown(f"**Architecture:** {model_info.get('architecture', 'N/A')}")
        st.markdown(f"**Training Date:** {model_info.get('training_date', 'N/A')}")
        st.markdown(f"**Accuracy:** {model_info.get('accuracy', 'N/A')}")
        st.markdown(f"**Description:** {model_info.get('description', 'No description available.')}")
    else:
        st.warning("No information available for this model.")

def create_probability_chart(probabilities):
    """Create and return a customized probability distribution chart"""
    # Ensure probabilities array matches FAULT_MAP length
    if len(probabilities) != len(FAULT_MAP):
        # Pad or truncate probabilities array to match FAULT_MAP length
        adjusted_probabilities = np.zeros(len(FAULT_MAP))
        for i in range(min(len(probabilities), len(FAULT_MAP))):
            adjusted_probabilities[i] = probabilities[i]
        probabilities = adjusted_probabilities
    
    prob_df = pd.DataFrame({
        "Fault Type": [FAULT_MAP[i]['name'] for i in range(len(FAULT_MAP))],
        "Probability": probabilities
    })
    
    # Set up custom style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Customize the style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
    
    # Create gradient colors
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(prob_df)))
    
    # Sort by probability for better visualization
    prob_df = prob_df.sort_values('Probability', ascending=True)
    
    # Plot horizontal bars with gradient colors
    bars = ax.barh(prob_df["Fault Type"], prob_df["Probability"], color=colors)
    
    # Customize the plot
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", fontsize=12, fontweight='bold')
    ax.set_title("Fault Probability Distribution", fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if width > 0.01:  # Only show labels for significant probabilities
            ax.text(max(width + 0.01, 0.05), bar.get_y() + bar.get_height()/2,
                    f'{width:.2%}', ha='left', va='center',
                    fontweight='bold', fontsize=10)
    
    # Customize grid and spines
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    return fig

def create_confusion_matrix_plot(confusion_matrix):
    """Create and return a customized confusion matrix plot"""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
    
    # Create heatmap with custom colormap and styling
    sns.heatmap(
        confusion_matrix, 
        annot=True, 
        fmt='d', 
        cmap='viridis',
        xticklabels=[FAULT_MAP[i]['name'] for i in range(len(FAULT_MAP))],
        yticklabels=[FAULT_MAP[i]['name'] for i in range(len(FAULT_MAP))],
        ax=ax,
        annot_kws={'size': 12, 'weight': 'bold'},
        cbar_kws={'label': 'Number of Samples'}
    )
    
    # Customize the plot
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    return fig

# FIXED: Improved with realistic default values based on provided measurements
def init_session_state():
    if 'ia' not in st.session_state:
        st.session_state.ia = -418.54  # Updated with realistic values from provided data
    if 'ib' not in st.session_state:
        st.session_state.ib = -49.02
    if 'ic' not in st.session_state:
        st.session_state.ic = 35.69
    if 'va' not in st.session_state:
        st.session_state.va = 0.27
    if 'vb' not in st.session_state:
        st.session_state.vb = -0.03
    if 'vc' not in st.session_state:
        st.session_state.vc = -0.25

# ADDED: Function to parse CSV data for batch testing
def parse_csv_data(csv_text):
    """Parse CSV data provided as text and extract measurement values"""
    try:
        lines = [line.strip() for line in csv_text.strip().split('\n') if line.strip()]
        data = []
        
        for line in lines:
            if ',' in line:
                values = line.split(',')
                if len(values) >= 7:  # First value could be fault label/class
                    # Extract the 6 measurement values (ignore any fault label)
                    measurements = [float(v) for v in values[-6:]]
                    # If the first values are binary (0/1), they might be one-hot encoded fault type
                    
                    fault_type = None
                    if all(v in ['0', '1'] for v in values[:4]):
                        fault_encoding = [int(v) for v in values[:4]]
                        fault_map = {
                            '0000': 0,  # No Fault
                            '1001': 1,  # LG
                            '0011': 2,  # LL
                            '0110': 3 , # LL
                            '1011': 4,  # LLG
                            '0111': 5,  # LLL
                            '1111': 6   # LLLG
                        }
                        # Convert from one-hot encoding to fault type if possible
                        if fault_encoding == [0, 0, 0, 0]:
                            fault_type = 0
                        elif fault_encoding == [1, 0, 0, 1]:
                            fault_type = 1
                        elif fault_encoding == [0, 0, 1, 1]:
                            fault_type = 2
                        elif fault_encoding == [0, 1, 1, 0]:
                            fault_type = 3
                        elif fault_encoding == [1, 0, 1, 1]:
                            fault_type = 4
                        elif fault_encoding == [0, 1, 1, 1]:
                            fault_type = 5
                        elif fault_encoding == [1, 1, 1, 1]:
                            fault_type = 6
                            # This might be a double-fault like LLG
                            # if fault_encoding == [0, 1, 1, 0]:
                            #     fault_type = 4  # LLG
                    
                    data.append({
                        'measurements': measurements,
                        'fault_type': fault_type
                    })
        
        return data
    except Exception as e:
        st.error(f"Error parsing CSV data: {str(e)}")
        return []

# Streamlit app
def main():
    st.set_page_config(
        page_title="Power System Fault Analyzer", 
        page_icon="⚡",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Load available models
    available_models = get_available_models()
    # print(available_models)
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Model type and selection
    if available_models:
        # Group models by type
        dl_models = {name: info for name, info in available_models.items() if info['type'] == 'Deep Learning'}
        # print(dl_models)
        ml_models = {name: info for name, info in available_models.items() if info['type'] == 'Machine Learning'}
        
        # Model type selection
        model_type = st.sidebar.radio(
            "Select Model Type",
            ["Deep Learning", "Machine Learning"],
            index=0
        )
        
        # Filter models based on type
        model_dict = dl_models if model_type == "Deep Learning" else ml_models
        available_model_names = list(model_dict.keys())
        
        if available_model_names:
            selected_model = st.sidebar.selectbox(
                "Select Model",
                available_model_names,
                index=0
            )
            model_path = available_models[selected_model]["path"]
            model = load_trained_model(model_path)
            model_info = available_models[selected_model]["info"]
        else:
            st.sidebar.warning(f"No {model_type} models available.")
            st.warning("No models found for the selected type. Please select a different model type or check your model directories.")
            model = None
            model_info = {}
    else:
        st.error("No models found in the configured directories! Please check the MODEL_DIRS configuration.")
        # Provide a dummy implementation for development/testing purposes
        st.warning("Running in demo mode with limited functionality.")
        model = None
        model_info = {}
    
    # Main content
    st.title("⚡ Power System Fault Analyzer")
    st.markdown("""
    This advanced tool detects and classifies faults in electrical power systems using machine learning models.
    Enter the current and voltage measurements to analyze potential faults.
    """)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Fault Detection", "Batch Analysis", "Model Information", "Fault Reference"])
    
    with tab1:
        st.header("Real-time Fault Detection")
        
        # Input section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Measurements (A)")
            ia = st.number_input("Ia (Phase A Current)", 
                                value=st.session_state.ia, 
                                min_value=-1000.0, 
                                max_value=1000.0,
                                step=0.1,
                                format="%.8f",  # FIXED: More practical precision
                                key="ia_input")
            ib = st.number_input("Ib (Phase B Current)", 
                                value=st.session_state.ib, 
                                min_value=-1000.0, 
                                max_value=1000.0,
                                step=0.1,
                                format="%.8f",
                                key="ib_input")
            ic = st.number_input("Ic (Phase C Current)", 
                                value=st.session_state.ic, 
                                min_value=-1000.0, 
                                max_value=1000.0,
                                step=0.1,
                                format="%.8f",
                                key="ic_input")
        
        with col2:
            st.subheader("Voltage Measurements (V)")
            va = st.number_input("Va (Phase A Voltage)", 
                                value=st.session_state.va, 
                                min_value=-1000.0, 
                                max_value=1000.0,
                                step=0.01,
                                format="%.8f",
                                key="va_input")
            vb = st.number_input("Vb (Phase B Voltage)", 
                                value=st.session_state.vb, 
                                min_value=-1000.0, 
                                max_value=1000.0,
                                step=0.01,
                                format="%.8f",
                                key="vb_input")
            vc = st.number_input("Vc (Phase C Voltage)", 
                                value=st.session_state.vc, 
                                min_value=-1000.0, 
                                max_value=1000.0,
                                step=0.01,
                                format="%.8f",
                                key="vc_input")
        
        # Update session state with current values
        st.session_state.ia = ia
        st.session_state.ib = ib
        st.session_state.ic = ic
        st.session_state.va = va
        st.session_state.vb = vb
        st.session_state.vc = vc
        
        # Create input array
        input_data = [ia, ib, ic, va, vb, vc]
        
        # Prediction button
        if st.button("⚡ Analyze System", type="primary", use_container_width=True):
            with st.spinner("Analyzing measurements..."):
                if model is not None:
                    predicted_class, confidence, probabilities, validation_metrics = predict_fault(
                        model, 
                        input_data
                    )
                    
                    # Display results
                    result_col1, result_col2 = st.columns([3, 1])
                    
                    with result_col1:
                        st.success(f"🔍 **Detection Result:** {FAULT_MAP[predicted_class]['name']}")
                    
                    with result_col2:
                        st.metric("Confidence Level", f"{confidence*100:.2f}%")
                    
                    # Display validation results if in validation mode
                    # if validation_mode and validation_metrics:
                    #     st.subheader("Validation Results")
                    #     validation_col1, validation_col2 = st.columns(2)
                        
                    #     with validation_col1:
                    #         st.info(f"True Fault Type: {FAULT_MAP[validation_metrics['true_class']]['name']}")
                    #         prediction_correct = validation_metrics['correct']
                    #         if prediction_correct:
                    #             st.success("✅ Prediction is correct!")
                    #         else:
                    #             st.error("❌ Prediction is incorrect")
                        
                    #     with validation_col2:
                    #         st.metric(
                    #             "Prediction Accuracy",
                    #             "100%" if prediction_correct else "0%",
                    #             delta="Correct" if prediction_correct else "Incorrect"
                    #         )
                    
                    # Probability distribution
                    st.subheader("Fault Probability Distribution")
                    fig = create_probability_chart(probabilities)
                    st.pyplot(fig)
                    
                    # Detailed fault information
                    display_fault_info(predicted_class)
                else:
                    st.error("Unable to perform analysis. Please check model selection.")
    
    # ADDED: New batch analysis tab
    with tab2:
        st.header("Batch Analysis")
        st.markdown("""
        Analyze multiple measurements at once by pasting CSV data below. 
        Each row should contain measurements in the format:
        `[optional_fault_type_encoding], ia, ib, ic, va, vb, vc`
        """)
        
        csv_data = st.text_area(
            "Paste measurement data in CSV format:", 
            height=200,
            help="Each line should have measurements in the format: [fault_encoding], ia, ib, ic, va, vb, vc"
        )
        
        print("\n " , csv_data)
        if st.button("Analyze Batch Data", type="primary"):
            if not csv_data.strip():
                st.warning("Please enter some measurement data to analyze.")
            else:
                parsed_data = parse_csv_data(csv_data)
                print("\n " , parsed_data)
                if parsed_data and model is not None:
                    st.success(f"Successfully parsed {len(parsed_data)} data points")
                    
                    # Create a table of results
                    results = []
                    for i, data_point in enumerate(parsed_data):
                        measurements = data_point['measurements']
                        true_fault = data_point['fault_type']
                        predicted_class, confidence, _, _ = predict_fault(
                            model, 
                            measurements,
                            validate=true_fault is not None,
                            true_class=true_fault
                        )
                        
                        results.append({
                            'Sample': i+1,
                            'Measurements': str(measurements),
                            'Predicted Fault': FAULT_MAP[predicted_class]['name'],
                            'Confidence': f"{confidence*100:.2f}%",
                            'True Fault': FAULT_MAP.get(true_fault, {}).get('name', 'Unknown') if true_fault is not None else 'Unknown',
                            'Correct': 'Yes' if true_fault == predicted_class else 'No' if true_fault is not None else 'Unknown'
                        })
                    
                    # Display results table
                    st.subheader("Batch Analysis Results")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Calculate and display accuracy metrics if true fault types are available
                    true_faults_available = any(data['fault_type'] is not None for data in parsed_data)
                    if true_faults_available:
                        correct_predictions = sum(1 for r in results if r['Correct'] == 'Yes')
                        total_with_truth = sum(1 for r in results if r['Correct'] != 'Unknown')
                        accuracy = correct_predictions / total_with_truth if total_with_truth > 0 else 0
                        
                        st.metric("Batch Accuracy", f"{accuracy*100:.2f}%", f"{correct_predictions}/{total_with_truth} correct")
                else:
                    st.error("Failed to parse data or model is not available.")
    
    with tab3:
        display_model_info(selected_model if 'selected_model' in locals() else "No model selected", model_info)
        
        # Additional model statistics
        if model_info:
            st.subheader("Performance Metrics")
            if model_info.get("metrics"):
                metrics = model_info["metrics"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{metrics.get('accuracy', 'N/A')}")
                col2.metric("Precision", f"{metrics.get('precision', 'N/A')}")
                col3.metric("Recall", f"{metrics.get('recall', 'N/A')}")
                
                # Confusion matrix if available
                if "confusion_matrix" in model_info:
                    st.subheader("Confusion Matrix")
                    cm = np.array(model_info["confusion_matrix"])
                    fig = create_confusion_matrix_plot(cm)
                    st.pyplot(fig)
            else:
                st.info("No performance metrics available for this model. Upload model metrics to the model_info.json file to see detailed performance data.")
        else:
            st.info("Select a model to view detailed information.")
    
    with tab4:
        st.header("Fault Type Reference Guide")
        
        selected_fault = st.selectbox(
            "Select a fault type to learn more:",
            [f"{i} - {FAULT_MAP[i]['name']}" for i in range(len(FAULT_MAP))]
        )
        fault_class = int(selected_fault.split(" - ")[0])
        
        display_fault_info(fault_class)
        
        # Add fault visualization placeholder
        st.subheader("Typical Waveform Pattern")
        st.info("Waveform visualizations will be available in future updates. Upload fault waveform images to enable this feature.")
        
        
        # Fault statistics (placeholder data - can be replaced with actual statistics)
        st.subheader("Industry Statistics")
        if fault_class == 0:
            st.markdown("""
            - **Normal Operation:** Represents ideal system conditions
            - **Occurrence:** Expected during >99% of system operation time
            - **Impact:** No damage or interruption to service
            """)
        elif fault_class == 1:  # LG
            st.markdown("""
            - **Frequency:** ~70% of all power system faults
            - **Clearance Time:** Typically 3-5 cycles
            - **Protection:** Usually cleared by ground overcurrent relays
            """)
        elif fault_class == 2:  # LL
            st.markdown("""
            - **Frequency:** ~15% of all power system faults
            - **Clearance Time:** Typically 3-6 cycles
            - **Protection:** Usually cleared by phase overcurrent or distance relays
            """)
        elif fault_class == 3:  # LL
            st.markdown("""
            - **Frequency:** ~15% of all power system faults
            - **Clearance Time:** Typically 3-6 cycles
            - **Protection:** Usually cleared by phase overcurrent or distance relays
            """)
        elif fault_class == 4:  # LLG
            st.markdown("""
            - **Frequency:** ~10% of all power system faults
            - **Clearance Time:** Typically 4-7 cycles
            - **Protection:** Usually cleared by both ground and phase protection
            """)
        elif fault_class == 5:  # LLL
            st.markdown("""
            - **Frequency:** ~3% of all power system faults
            - **Clearance Time:** Typically 2-4 cycles
            - **Protection:** Usually cleared by differential or distance protection
            """)
        elif fault_class == 6:  # LLLG
            st.markdown("""
            - **Frequency:** ~2% of all power system faults
            - **Clearance Time:** Typically 2-3 cycles
            - **Protection:** Usually cleared by main differential protection
            """)
    
    # Footer with sample test data
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Test Samples")
    
    def set_sample_values(values):
        st.session_state.ia = values[0]
        st.session_state.ib = values[1]
        st.session_state.ic = values[2]
        st.session_state.va = values[3]
        st.session_state.vb = values[4]
        st.session_state.vc = values[5]
    
    # Sample test data for different fault types
    samples = {
        "No Fault": [1.809045629,85.58214872,-90.72164987,-0.575582481,0.501113559,0.074468922],
        "LG Fault": [-557.3918085,-119.4686428,-29.52944976,0.210003736,-0.076712453,-0.133291283],
        "LL Fault": [117.7762421,-40.14452634,49.6842013,0.366859393,-0.591493195,0.224633802],
        "LLG Fault": [325.3, 298.7, 1.2, 10.5, 8.9, 230.1],
        "Three-Phase Fault": [-703.0121029,-74.68896319,779.8781938,-0.017158282,0.029459464,-0.012301182],
        "LLLG Fault": [485.2, 502.7, 493.5, 0.7, 0.5, 0.6]
    }
    
    for name, values in samples.items():
        if st.sidebar.button(name, key=f"sample_{name}"):
            set_sample_values(values)
            st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Instructions:**
    1. Select a model from the dropdown
    2. Enter current and voltage measurements or use a quick test sample
    3. Click 'Analyze System'
    4. Review results and recommendations
    
    **Note:** For more accurate results, ensure model paths and directories are correctly configured.
    """)
    
    # Version information
    st.sidebar.markdown("---")
    st.sidebar.caption("Power System Fault Analyzer v1.0.1")
    st.sidebar.caption("© 2025 Power FaultSystems Engineering , P.F.E")

if __name__ == "__main__":
    main()