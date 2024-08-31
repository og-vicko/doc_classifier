## Document Classifier for Financial Law Relevance
--------------------------------------------------------------------------------------------------------------------

This project is a machine learning-based document classification system designed to determine the relevance of legal documents to financial law. The classifier processes legal documents and predicts whether they are relevant to financial law, streamlining the process of legal document review.

- Table of Contents
- Project Overview
- Features
- Installation
- Usage
- Model Details
- Evaluation Metrics
- Streamlit Deployment
- File Structure

##### Project Overview
The Document Classifier for Financial Law Relevance is designed to automate the process of identifying whether a legal document pertains to financial law. This system uses a machine learning pipeline that includes data preprocessing, feature engineering, and a classification model to predict document relevance. It can handle bulk document classification and is easily deployable via a Streamlit web application.

##### Features
Automated Document Classification: Automatically classify legal documents based on their relevance to financial law.
Bulk File Processing: Upload and process multiple documents at once.
Confidence Scores: Receive confidence scores along with the classification result.
Interactive Web Interface: User-friendly interface for document upload and prediction using Streamlit.
Downloadable Results: Option to download the classification results as a CSV file.

##### Installation
Prerequisites
Ensure that you have the following installed:

- Python 3.8+
    pip (Python package installer)
    Setup
    Clone the repository:

    git clone https://github.com/your-username/document-classifier.git
    cd document-classifier

- Create a virtual environment and activate it:
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
 
- Install the required dependencies:
    pip install -r requirements.txt
    
- Load the pre-trained model:
    Place the pre-trained model file (document_classifier.dill) in the models directory. The model should be trained and exported using the provided training scripts.

##### Usage
Local Prediction
To make predictions locally on a set of documents:

    Prepare your CSV file containing the documents with the required fields (e.g., title, content, publicationdate).


- Streamlit Web Application
    Run the Streamlit app:
        streamlit run app.py

Upload your CSV file through the web interface.

View the predictions on the web interface and download the results if needed.

##### Model Details
The classification model is a pipeline that includes:

Text Preprocessing: Handling missing values, text normalization, and feature engineering (e.g., combining title and content).
Feature Engineering: Week of publication, user and requirement source placeholders, and combined text features.
Classification Model: A machine learning model trained on labeled legal documents to predict relevance to financial law.
The model was trained using a diverse set of legal documents, and various evaluation metrics were used to assess its performance.

##### Evaluation Metrics
To evaluate the performance of the classification model, the following metrics were considered:

- Accuracy: Overall correctness of the model.
- Precision: Ability of the model to identify only relevant documents.
- Recall: Ability of the model to identify all relevant documents.
- F1-Score: Harmonic mean of precision and recall, providing a balance between the two.
- AUC-ROC: Measures the model's ability to distinguish between relevant and irrelevant documents.

#### Streamlit Deployment
The project includes a Streamlit web application for easy deployment and user interaction:

File Upload: Users can upload CSV files containing the documents to be classified.
Prediction: The app processes the uploaded documents and predicts their relevance.
Download: Users can download the predictions as a CSV file.

#### File Structure
finaincial_doc_labelling_pipeline/
│
|__ evaluation_metrics/
|   └── # evaluation plots .png files
|__ notebooks/
|   └── model building_1.ipynb # notebook to train model
├── deployment/
|   └── app.py      # Streamlit app for deployment
|   └── preprocess_functions.py # Preprocess text functions
├── predict.py            # Script for making local predictions
├── models/
│   └── document_classifier.dill  # Pre-trained model file and other models
├── requirements.txt      # Dependencies for the project
├── README.md             # Project documentation
└── data/                 # Example data files (optional)

License
This project is licensed under the MIT License - see the LICENSE file for details.

