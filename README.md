
# Startup Success Predictor

This project is a **Streamlit application** that predicts the success likelihood of a startup based on funding and company characteristics. The model is built with a **Random Forest Classifier** and enhanced with data preprocessing using **SMOTE** for balanced training.

## Features
- **Predictive Model**: The app uses a Random Forest Classifier to predict whether a startup is likely to succeed.
- **User Input Options**: Allows users to input startup information such as funding rounds, total funding, valuation, and whether it has received various types of funding (VC, Angel, etc.).
- **Probability Output**: Displays the prediction along with the success and failure probabilities.

## Files and Directories
- `app/app.py`: Main Streamlit app file.
- `model/model_with_scaler_and_encoder.pkl`: Contains the saved model, scaler, label encoder, and feature names.
- `notebook/workbook_1.ipynb`: Contains the working notebook
- `README.md`: Project overview (this file).

## Installation and Setup

### Prerequisites
- Python 3.7+
- Required Python packages (see `requirements.txt`)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/philippe2023/startup-success-predictor.git
    cd startup-success-predictor
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application
To start the app, use the following command in the terminal:
```bash
streamlit run app/app.py
