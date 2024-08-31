import joblib
import os
import streamlit as st
import pandas as pd
import numpy as np
import dill
import warnings
from preprocess_functions import *

warnings.filterwarnings('ignore')


st.title('DOCUMENT CLASSIFIER')


# def load_model_from_dir(model_name):
#     current_dir = os.getcwd()

#     # Navigate one level up to the parent directory
#     parent_dir = os.path.dirname(current_dir)
#     models_dir = os.path.join(parent_dir, 'models')
#     file_path = os.path.join(models_dir, model_name)
    
#     # Check if the model file exists
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"The model file {model_name} does not exist in the directory {models_dir}.")

#     # Load the model from the specified path using dill
#     try:
#         with open(file_path, 'rb') as f:
#             model = dill.load(f)
#         print(f"Model loaded from {file_path}")
#     except Exception as e:
#         raise RuntimeError(f"Failed to load model: {e}")
    
#     return model        
def load_model_from_dir(model_name):
    # Get the current script directory
    current_dir = os.path.dirname(__file__)
    # Define the path to the models directory
    models_dir = os.path.join(current_dir, '..', 'models')
    # Construct the full file path
    file_path = os.path.join(models_dir, model_name)
    
    # Check if the model file exists
    if not os.path.exists(file_path):
        st.error(f"The model file {model_name} does not exist in the directory {models_dir}.")
        return None
    
    # Load the model from the specified path using dill
    try:
        with open(file_path, 'rb') as f:
            model = dill.load(f)
        st.success(f"Model loaded from {file_path}")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
######LOAD MODEL#######
pipeline = load_model_from_dir('document_classifier.dill')


def week_of_month(date):
    date = pd.to_datetime(date)  # Ensure date is a datetime object
    first_day = date.replace(day=1)
    dom = date.day
    adjusted_dom = dom + first_day.weekday()  # Adjust day of month for the first day of the week
    return int(np.ceil(adjusted_dom / 7.0))


def make_predictions(df):
    """
    Make predictions using the trained pipeline and determine the relevance of each document.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the features needed for prediction.
        
    Returns:
        pd.DataFrame: DataFrame containing the predictions and their respective probabilities.
    """
    probabilities = pipeline.predict_proba(df)
    df['Probability Of Relevance'] = probabilities[:, 1]
    
    # Determine decision for each document
    df['Decision'] = df['Probability Of Relevance'].apply(lambda x: 'Relevant' if x > 0.5 else 'Irrelevant')
    
    return df[['Probability Of Relevance', 'Decision']]


# Upload File fro prediction
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])


def prepare_data(data):

    #Engineer columns to match shape of train data
    data['title_and_content'] = data['title'] + data['content']
    data.drop(['title','content'], axis=1, inplace=True)

    data['week'] = data['publicationdate'].apply(week_of_month)
    data['week'] = data['week'].astype('category')
    data['user'] = ''
    data['requirementsource'] = ''
    data['title_and_content'].fillna('Missing', inplace=True) 

    return data

# if uploaded_file is not None:
#     # Read the file into a DataFrame
#     df = pd.read_csv(uploaded_file)

#     prepare_df = prepare_data(df.head())
#     predictions = make_prediction(prepare_df)

#     st.success(predictions)
if uploaded_file is not None:
    with st.spinner("Processing..."):
        # Read the file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Prepare data and make predictions
        df = df.head()
        prepared_df = prepare_data(df)
        predictions_df = make_predictions(prepared_df)
        st.write(predictions_df)

        # Download button to download predictions
        csv = predictions_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download predictions as CSV",
                        data=csv,
                        file_name='predictions.csv',
                        mime='text/csv')

    st.success("Done!")