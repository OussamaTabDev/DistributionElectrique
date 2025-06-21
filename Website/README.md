# Web Interface for Electrical Distribution Fault Analysis

This directory contains the Streamlit-based web application for the Electrical Distribution Fault Analysis System. The web app provides an interactive interface for users to visualize, analyze, and classify faults in electrical distribution networks using the machine learning and deep learning models developed in this project.

## What is Streamlit?
[Streamlit](https://streamlit.io/) is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. With Streamlit, you can turn data scripts into interactive web applications in minutes, all in pure Python—no front-end experience required.

## How the Web App Relates to the Project
- The web app serves as the user-facing layer of the project, allowing users to:
  - Upload and analyze new data
  - Run trained models (from the `Deep Learning` and `Machine Learning` folders) to classify faults
  - Visualize model performance and results (confusion matrices, metrics, etc.)
  - Explore data distributions and patterns
- The app loads pre-trained models (e.g., `.h5` files for deep learning) and uses them to make predictions on user-supplied or test data.
- All heavy computation and model training are done offline (in the notebooks/scripts); the web app is for inference and visualization.

## Getting Started

1. **Install dependencies** (from the project root):
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser:**
   Streamlit will provide a local URL (usually http://localhost:8501) where you can interact with the app.

## Main Features
- **Model Selection:** Choose from multiple trained models for fault classification.
- **Data Upload:** Upload your own CSV files for analysis.
- **Prediction:** Get real-time predictions and see which class a sample belongs to.
- **Visualization:** View confusion matrices, performance metrics, and data distributions.
- **Documentation:** Access model info and usage instructions directly from the app.

## File Overview
- `app.py` — Main Streamlit application file.
- `model_info.json` — Metadata about available models.
- Pre-trained model files are loaded from the parent directories as needed.

## Tips for Newcomers to Streamlit
- Streamlit apps are written in Python scripts. You run them with `streamlit run <script.py>`.
- UI elements (buttons, file uploaders, charts) are created using simple Python functions (e.g., `st.button`, `st.file_uploader`, `st.pyplot`).
- The app reruns from top to bottom on every user interaction, so keep heavy computations outside the main script or cache them.
- You can quickly prototype and share data apps without any HTML, CSS, or JavaScript.

## Customization
- To add new models, update `model_info.json` and place the model files in the appropriate directory.
- To change the UI or add new features, edit `app.py` using Streamlit's API.

## Further Reading
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cheat Sheet](https://docs.streamlit.io/library/cheatsheet)

---
This web interface is an integral part of the Electrical Distribution Fault Analysis System, making advanced analytics accessible to engineers and researchers through an intuitive browser-based tool.
