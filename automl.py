import numpy as np
import pandas as pd
import streamlit as st
import os
# import streamlit.components.v1 as st_component
from PIL import Image
from Util import exploratory_data_analysis as eda
from Util import dataset_utils
from Util import missing_value as mv
from Util import Regression_Modelling as md
from Util import Classificatio_modeling as cm
from Util import prediction as pred
from Util import outlier as out
from Util import cv_modeling as cv
from Util import prediction
from Util import preprocess as pp
# from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime
# from streamlit_drawable_canvas import st_canvas
from streamlit_image_annotation import detection
# import zipfile
# import yaml
from pathlib import Path
# import random
import shutil
import io

pageicon = Image.open(r"Images/3i_icon.png") 
st.set_page_config(page_title="FutureTech AutoML",
                    page_icon=pageicon,
                    layout='wide')
# Page Title
title_col1, title_col2, title_col3 = st.columns([0.4, 0.5, 0.2])
with title_col2:
    st.title("*FutureTech AutoML*")

st.title("")
st.title("")
st.title("")

if 'create_dataset' not in st.session_state:
    st.session_state.create_dataset = False

if 'analyze_dataset' not in st.session_state:
    st.session_state.analyze_dataset = False

if 'training' not in st.session_state:
    st.session_state.training = False

if 'preprocessing' not in st.session_state:
    st.session_state.preprocessing = False

if 'preprocessed_preview' not in st.session_state:
    st.session_state.preprocessed_preview = False

if 'preprocessed_download' not in st.session_state:
    st.session_state.preprocessed_download = False

if 'start_training' not in st.session_state:
    st.session_state.start_training = False
    
if 'prediction' not in st.session_state:
    st.session_state.prediction = False

if 'predict' not in st.session_state:
    st.session_state.predict = False
    
if 'auto_eda' not in st.session_state:
    st.session_state.auto_eda = False
    
if 'advanced_auto_eda' not in st.session_state:
    st.session_state.advanced_auto_eda = False
    
if 'create_eda' not in st.session_state:
    st.session_state.create_eda = False
    
if 'create_charts' not in st.session_state:
    st.session_state.create_charts = False
    

def click_create_dataset():
    st.session_state.training = False
    st.session_state.prediction = False
    st.session_state.create_dataset = True
    
def click_analyze_dataset():
    st.session_state.create_dataset = False
    st.session_state.analyze_dataset = True
    
def click_training():
    st.session_state.prediction = False
    st.session_state.create_dataset = False
    st.session_state.training = True

def click_create_experiment():
    st.session_state.create_experiment = True
    
def click_preprocessing():
    st.session_state.start_training = False
    st.session_state.preprocessed_preview = False
    st.session_state.preprocessed_download = False
    st.session_state.preprocessing = True
        
def click_preprocessed_preview():
    st.session_state.preprocessed_download = False
    st.session_state.preprocessed_preview = True
    
def click_preprocessed_download():
    st.session_state.preprocessed_preview = False
    st.session_state.preprocessed_download = True

def click_start_training():
    st.session_state.start_training = True

def click_prediction():
    st.session_state.create_dataset = False
    st.session_state.training = False
    st.session_state.prediction = True

def click_predict():
    st.session_state.predict = True
    
def click_auto_eda():
    st.session_state.auto_eda = True

def click_advanced_auto_eda():
    st.session_state.advanced_auto_eda = True

def click_create_eda():
    st.session_state.auto_eda = False
    st.session_state.advanced_auto_eda = False
    st.session_state.create_eda = True

def click_create_charts():
    st.session_state.create_charts = True

@st.cache_data
def save_experiment_details(name_of_experiment, date_and_time, ml_type, dataset_name, task=None, target=None, use_case_flag=None):
    details = f"Experiment Name: {name_of_experiment}\nTime & Date of Experiment Creation: {date_and_time}\nML Type: {ml_type}\nTasks: {task}\nInput Dataset Name: {dataset_name}\nTarget Column: {target}\nUse Case_Flag: {use_case_flag}"

    with open(Path(os.path.join('Experiments', name_of_experiment, 'exp_details.txt')), "w") as file:
        file.write(details)

@st.cache_data
def extract_experiment_details(file_path):
    
    experiment_name = None
    creation_time = None
    dataset_name = None
    target_column = None
    task = None
    use_case_flag = None
    
    with open(file_path, "r") as file:
        for line in file:
            key, value = line.strip().split(": ")

            if key == "Experiment Name":
                experiment_name = value
            elif key == "Time & Date of Experiment Creation":
                creation_time = value
            elif key == "ML Type":
                ml_type = value
            elif key == "Input Dataset Name":
                dataset_name = value
            elif key == "Target Column":
                target_column = value
            elif key == "Tasks":
                task = value
            elif key == "Use Case_Flag":
                use_case_flag = value
                
    return experiment_name, creation_time, ml_type, dataset_name, target_column, task, use_case_flag
   
# @st.cache_data(experimental_allow_widgets=True)
def main():            
    button_col1, button_col2, button_col3 = st.columns(3)
    # Dataset Upload Option
    with button_col1:
        col1, col2 = st.columns([0.1, 0.9])
        with col2:
            st.subheader("Prepare Your Dataset")
        st.write("Collect and prepare your data to train the model")
        col1, col2 = st.columns([0.3, 0.7])
        with col2:
            create_dataset_button = st.button("Create Dataset", on_click=click_create_dataset)

    # #Exploratory Data Analysis Button
    # with button_col2:
    #     col1, col2 = st.columns([0.2, 0.8])
    #     with col2:
    #         st.subheader("Analyze Your Data")
    #     st.write("Do exploratory data analysis on your data to get hidden and meaningful inferances from it.")
    #     col1, col2 = st.columns([0.25, 0.75])
    #     with col2:
    #         analyze_button = st.button("Anlyze Dataset", on_click=click_analyze_dataset)
            
    # Create Model Option
    with button_col2:
        col1, col2 = st.columns([0.2, 0.8])
        with col2:
            st.subheader("Train Your Model")
        st.write("Train the best-in-class machine learning model with your dataset")
        col1, col2 = st.columns([0.25, 0.75])
        with col2:
            train_model_button = st.button("Start Model Training", on_click=click_training)
        
    # Prediction Option
    with button_col3:
        col1, col2 = st.columns([0.2, 0.8])
        with col2:
            st.subheader("Get Predictions")
        st.write("After you train the model, you can use it to get predictions, either online or batch.")
        col1, col2 = st.columns([0.35, 0.65])
        with col2:
            prediction_button = st.button("Prediction", on_click=click_prediction)
    
    
    # Dataset Upload        
    if st.session_state.create_dataset:
        "---"
        title_col1, title_col2= st.columns([0.42, 0.58])
        with title_col2:
            st.subheader("Create Dataset")
            
        col1, col2 = st.columns(2)
        with col1:
            # Datset Name            
            dataset_name = st.text_input("Dataset Name", max_chars=100)
            # Dataset Type
            data_type = st.selectbox("Select Data Type", options=['Tabular', 'Text', 'Image', 'Video'], help="Select the type of data your datset will contain")
            # Dataset Source
            data_source_options = ["Upload file from your computer", "Upload file from cloud storage", "Select table or view from database", 'Upload through url', 'Live Feed through WebCam']      
            data_source = st.radio("Select Data Source", options=data_source_options)
        
        with col2:
            # Dataset Descriptiom
            dataset_desc = st.text_area("Description", height=120, max_chars=500)
            # Dataset File
            if data_source == "Upload file from your computer":
                input_data = st.file_uploader("Upload data file from your computer", type=None)
            
            elif data_source == "Upload through url":
                input_data = st.text_input("Enter the image URL:")
                
            elif data_source == "Live Feed through WebCam":
                pass

            else:
                st.error("Development under progress for this DataType")
        
        if data_source != "Live Feed through WebCam":
            if not dataset_utils.upload_checker(dataset_name=dataset_name, data_type=data_type, input_data=input_data):
                global data
                data = dataset_utils.make_dataset(data_source, input_data, dataset_name, data_type)

                if data is not None:
                    st.success("Dataset Creation Completed.")
        else:
            st.success("Dataset Creation Completed.")

            # data = dataset_utils.fetch_dataframe()
        
        if data_type == 'Tabular':
            st.title("")
            
            col1, col2, col3, col4, col5, col6, col7 = st.columns([0.2, 0.1, 0.15, 0.15, 0.12, 0.12, 0.15])
            # Data Preview Button
            with col2:    
                preview_button = st.button("Preview")
            # Auto EDA Button
            with col3:
                auto_eda_button = st.button("Get EDA Report", on_click=click_auto_eda)
            # Manual EDA Button
            with col4:
                manual_eda_button = st.button("Create EDA Report", on_click=click_create_eda)
            # Quick Training Button   
            with col5:
                quick_training_button = st.button("Quick Train", key="quick_train1")
            # Training Button   
            with col6:
                training_button = st.button("Train Model", key="train1")
            
            # Dataset Preview 
            if preview_button:
                st.title('')
                dataset_utils.get_dataset_preview(data)
                    
            # Automated Exploratory Data Analysis    
            elif st.session_state.auto_eda == True:
                # if dataset_utils.upload_checker():
                #     error = dataset_utils.upload_checker()
                #     st.write(f":red[{error}]")          
                # else:
                st.title('')
                eda.automated_eda_report(data)
                
                # Advanced EDA Button
                advanced_eda_button = st.button("Show Advanced EDA", on_click=click_advanced_auto_eda)
                if st.session_state.advanced_auto_eda == True:
                    eda.advanced_eda(data)
                    
                st.title("")
                
                col1, col2, col3, col4 = st.columns([0.37, 0.15, 0.15, 0.33])
                with col2:
                    quick_training_button = st.button("Quick Train", key="quick_train2")
                # Training Button   
                with col3:
                    training_button = st.button("Train Model", key="train2")
                    
            # Manual Exploratory Data Analysis    
            elif st.session_state.create_eda == True:
                st.session_state.auto_eda = False
                if dataset_utils.upload_checker(dataset_name, data_type, input_data):
                    error = dataset_utils.upload_checker(dataset_name, data_type, input_data)
                    st.write(f":red[{error}]")          
                else:
                    "---"
                    col1, col2, col3 = st.columns([0.33, 0.4, 0.3])
                    with col2:
                        st.header("**Exploratory Data Analysis**")
                    st.subheader("")
                    
                    col1, col2, col3, col4, col5, col6, col7 = st.columns([0.15, 0.12, 0.18, 0.12, 0.12, 0.15, 0.15])
                    with col2:
                        summary_eda_button = st.button("Summary")
                    with col3:
                        missing_val_eda_button = st.button("Missing Value Analysis")
                    with col4:
                        corr_matrics_button = st.button("Correlation")
                    with col5:
                        stats_button = st.button("Statistics")
                    with col6:
                        st.button("Create Charts", on_click=click_create_charts)
                    
                    # Dataset Summary
                    if summary_eda_button:
                        st.title('')
                        col1, col2, col3 = st.columns([0.4, 0.4, 0.2])
                        with col2:
                            st.subheader("Dataset Summary")
                        eda.get_dataset_summary(data)
                    
                    # Missing Value analysis
                    elif missing_val_eda_button:
                        st.title('')
                        eda.get_missing_val_analysis(data)
                        
                    # Corelation Matrics
                    elif corr_matrics_button:
                        st.title('')
                        eda.get_correlation_matrics(data)
                    
                    # Discriptive Statistics
                    elif stats_button:
                        st.title('')
                        col1, col2, col3 = st.columns([0.38, 0.4, 0.2])
                        with col2:
                            st.subheader("Discriptive Statistics")
                        eda.get_discriptive_stats(data)
                        
                    # Charts & Graphs
                    elif st.session_state.create_charts:
                        st.title('')
                        col1, col2, col3, col4, col5 = st.columns([0.2, 0.2, 0.2, 0.2, 0.2])
                        with col2:
                            x_col = st.selectbox("X-Axis", options=data.columns)
                        with col3:
                            y_col = st.selectbox("Y-Axis", options=data.columns)
                        with col4:
                            chart = st.selectbox("Chart", options=['Line', 'Bar', 'Scatter', 'Pie', 'Heatmap', 'Histplot', 'Countplot'])
                                            
                        col1, col2, col3, col4 = st.columns([0.3, 0.2, 0.2, 0.3])
                        with col2:
                            if chart == 'Heatmap':
                                val_col = st.selectbox("Values", options=data.columns)
                            elif chart == 'Pie':
                                pass
                            else:
                                grp_options = ['']
                                grp_options.extend(data.columns.to_list())
                                grp_col = st.selectbox("Group By", options=grp_options)
                        with col3:
                            if chart == 'Countplot':
                                aggregator = st.selectbox("Aggregation", options=['count', 'percent', 'proportion', 'probability'])
                            elif chart == 'Histplot':
                                aggregator = st.selectbox("Aggregation", options=['density', 'count', 'percent', 'proportion', 'probability', 'frequency'])
                            elif chart == 'Scatter':
                                size_options = ['']
                                size_options.extend(data.columns.to_list())
                                size_col = st.selectbox("Size By", options=size_options)
                            elif chart == 'Pie':
                                pass
                            else:              
                                aggregator = st.selectbox("Aggregation", options=['', 'mean', 'median', 'sum', 'count', 'max', 'min'])
                                                                    
                        col1, col2, col3 = st.columns([0.4, 0.2, 0.3])
                        with col2:
                            show_button = st.button("Show Chart")
                            
                        if show_button:
                            if chart in ['Histplot', 'Countplot', 'Pie']:
                                if x_col != y_col:
                                    st.write(":green[For this type of chart only one column is needed. Keep both X-axis and Y-axis Same]")
                                else:
                                    if chart == 'Pie':
                                        eda.get_pieplot(data, x_col)
                                    elif chart == 'Histplot':
                                        eda.get_histplot(data, x_col, grp_col, aggregator)
                                    else:
                                        eda.get_countplot(data, x_col, grp_col, aggregator)              
                            else:
                                if chart == 'Line':
                                    eda.get_lineplot(data, x_col, y_col, grp_col, aggregator)
                                elif chart == 'Scatter':
                                    eda.get_scatterplot(data, x_col, y_col, grp_col, size_col)
                                elif chart == 'Bar':
                                    eda.get_barplot(data, x_col, y_col, grp_col, aggregator)
                                elif chart == 'Heatmap':
                                    eda.get_heatmap(data, x_col, y_col, val_col, aggregator)   

    if st.session_state.training:        
        "---"
        col1, col2, col3 = st.columns([0.4, 0.3, 0.3])
        with col2:
            st.header("Model Training")
                
        col2, col2, col3, col4, col5, col6 = st.columns([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
        with col2:
            exp_name = st.text_input("Experiment Name")
        with col3:
            ml_type = st.selectbox('ML Problem Type', options=['Regression',
                                                               'Classification',
                                                               'Forecasting',
                                                               'Natural Language Processing',
                                                               'Computer Vision'])
        with col4:
            if ml_type in ['Regression', 'Classification', 'Forecasting']:
                dataset_list = ['']
                dataset_list.extend(os.listdir("Datasets"))  #if file.endswith('.csv')
                dataset_name = st.selectbox('Dataset', options=dataset_list)
                if dataset_name != '':
                    df = pd.read_csv(Path(os.path.join("Datasets", dataset_name, f"{dataset_name}.csv")))
                    target_option = df.columns.tolist()
                else:
                    target_option = ['']
                with col5:
                    global target 
                    target= st.selectbox('Target', options=target_option)
            
            elif ml_type == 'Computer Vision':
                task_list = ['Image_Classification',
                             'Image_Segmentation',
                             'Object_Detection_through_Image',
                             'Object_Detection_through_Video']
                
                target = st.selectbox('Task', options=task_list)
                
                with col5:
                    dataset_list = ['Live Feed through WebCam']
                    dataset_list.extend([file for file in os.listdir("Datasets")]) #  if file.endswith('.mp4')
                    dataset_name = st.selectbox('Dataset', options=dataset_list)
                                        
        st.title("")
        
        if not md.experiment_checker(exp_name, dataset_name, ml_type, target):
            current_datetime = datetime.now()
            date_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            
            os.makedirs(Path(os.path.join('Experiments', exp_name)), exist_ok=True)
            
            if ml_type in ['Regression', 'Classification', 'Forecasting']:
                save_experiment_details(exp_name, date_time, ml_type, dataset_name, target=target)
                
                col1,col2,col3,col4= st.columns(4)
                with col1:
                    st.subheader("Feature Selection")
                    #df= df_original.copy()
                    #
                    df_data = pd.DataFrame({
                        'Column_Name': df.columns,
                        'Data_Type': df.dtypes.values
                    })
                    df_with_selections = df_data.copy()
                    df_with_selections.insert(0, "Include", False)
                    df_with_selections['Include'] = df_with_selections['Column_Name'] == target
                    
                    # Get dataframe row-selections from user with st.data_editor
                    
                    edited_df = st.data_editor(
                        df_with_selections, 
                        hide_index=True,
                        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
                        disabled=df.columns,
                        num_rows="dynamic",
                    )

                    # Filter the dataframe using the temporary column, then drop the column
                    selected_rows = edited_df[edited_df.Include]
                    if (edited_df['Include']== 1).any():
                        selected_rows.drop('Include', axis=1)
                        col_list = selected_rows['Column_Name'].tolist()

                    with open(Path(os.path.join("Experiments", exp_name, "selected_cols.pkl")), 'wb') as f:
                        pickle.dump(col_list, f)
                        
                # Missing Value Treatment
                with col2:
                    st.subheader("Missing Value Treatment")
                    df_mv = df[col_list]
                    # Get columns and their data types
                    column_data_types = mv.get_column_data_types(df_mv)
                    # Create a new column for Imputation Method in DataFrame
                    column_data_types['Imputation Method'] = [''] * len(column_data_types)
                    # Display dropdowns for selecting imputation method for each column   
                    for _, row in column_data_types.iterrows():
                        # if row['Column Name'] != 'ADDRESS':
                        col = row['Column Name']
                        data_type = row['Data Type']
                        # Customize dropdown options based on data type
                        if data_type == 'object':
                            imputation_methods = ['auto', 'most_frequent', 'classification']  # Add more methods as needed for object type
                        else:
                            imputation_methods = ['auto', 'mean', 'most_frequent', 'median', 'max', 'min', 'regression', 'KNN']  # Add more methods as needed for numeric type

                        imputation_method = st.selectbox(f'Select Imputation Method for {col} ({data_type}):', imputation_methods,
                                                        key=col)

                        # Update the DataFrame with the selected imputation method
                        column_data_types.loc[column_data_types['Column Name'] == col, 'Imputation Method'] = imputation_method
                #pp.col_include(df,col1,col2,col3,col4,target)
                
                # Outlier Treatment
                with col3:
                    st.subheader("Outlier Treatment")
                    
                    # st.dataframe(df)
                    # st.title('Treating Outliers Configuration')

                    # Get columns and their data types
                    columns_with_outliers = out.detect_outliers(df_mv)

                    # Create a new column for Imputation Method in DataFrame
                    # Convert the list to a DataFrame
                    outliers_df = pd.DataFrame(columns_with_outliers, columns=['Column Name'])
                    outliers_df['Treatment Method'] = [''] * len(outliers_df)
                    # Display table with columns, data types, and dropdowns
                    # st.write('### DataFrame with Treatment Configuration:')

                    # Display dropdowns for selecting imputation method for each column

                    for i,row in outliers_df.iterrows():

                        col = row['Column Name']
                        treatment_methods = ['auto', 'IQR', 'zscore','svm','knn','isolation_forest']  # Add more methods as needed for object type
                        treatment_method = st.selectbox(f'Select Treatment Method for {col} :', treatment_methods, key=col + str(i))

                        # Update the DataFrame with the selected imputation method
                        outliers_df.loc[outliers_df['Column Name'] == col, 'Treatment Method'] = treatment_method

                with col4:
                    st.subheader("Feature Encoding")

                    categorical_columns = df_mv.select_dtypes(include=['object']).columns 
                    column_cat_types= df_mv[categorical_columns].dtypes.reset_index().rename(columns={0: 'Data Type', 'index': 'Column Name'})

                    # Create a new column for Imputation Method in DataFrame
                    column_cat_types['Encoding Method'] = [''] * len(column_cat_types)
                    
                    # Display dropdowns for selecting imputation method for each column   
                    for i, row in column_cat_types.iterrows():
                        # if row['Column Name'] != 'ADDRESS':
                        col = row['Column Name']
                        data_type = row['Data Type']
                        
                        # Customize dropdown options based on data type
                        encoding_methods = ['One-Hot Encoding', 'Label Encoding', 'Ordinal Encoding', 'Frequency Encoding']  # Add more methods as needed for object type  
                        selected_encoding = st.selectbox(f'Select Encoding Method for {col} ({data_type}):', encoding_methods, key=col + str(i+1))
                        column_cat_types.loc[column_cat_types['Column Name'] == col, 'Encoding Method'] = selected_encoding 


                    # Feature Scaling
                    st.subheader("Feature scaling")
                    scaler = st.selectbox("Select scaling Method:",['Min-max Scaling','Standarization','Robust Scaling'])
                    # Update the DataFrame with the selected imputation method
                    # column_data_types.loc[column_data_types['Column Name'] == col, 'Encoding Method'] = scaler 
                
                st.title("")
                col1, col2, col3, col4, col5, col6 = st.columns([0.1, 0.2, 0.2, 0.2, 0.2, 0.05])
                with col2:
                    st.button("Do the Preprocessing", on_click=click_preprocessing)
                
                    if st.session_state.preprocessing:
                        st.title("")
                        imputed_df= pp.missing_value(df_mv,column_data_types)
                        treated_df= pp.outlier_treatment(imputed_df,outliers_df)
                        encoded_df= pp.encode_categorical_columns(treated_df,column_cat_types)
                        X= encoded_df.drop(target,axis=1)
                        y= encoded_df[[target]]
                        
                        scaler, scaled_df=pp.scaling(encoded_df,X,y,scaler)
                        
                        with open(Path(os.path.join("Experiments", exp_name, "scaler.pkl")),'wb') as f:
                            pickle.dump(scaler,f)
                            
                        scaled_df.to_csv(Path(os.path.join("Experiments", exp_name, "Transformed_df.csv")))
                
                if st.session_state.preprocessing:
                    with col3:
                        st.button("Preview Preprocessed Dataset", on_click=click_preprocessed_preview)
                    
                    with col4:
                        st.download_button("Download Preprocessed Dataset", data=scaled_df.to_csv().encode('utf-8'), file_name='preprocessed_df.csv', mime='text/csv', on_click=click_preprocessed_download)
                            
                    with col5:
                        st.button("Start Training", on_click=click_start_training)
                 
                if st.session_state.preprocessed_preview:
                    st.subheader('Preprocessed Dataset')
                    st.dataframe(scaled_df.head(10))
                
                if st.session_state.start_training:  
                    # exp_list = ['']
                    # exp_list.extend([file.split('.')[0] for file in os.listdir("Experiments") if file.endswith('.csv')])
                    Regression_model_list = ["Random Forest Regression","Linear Regression","Ridge Regression","Lasso Regression","ElasticNet", "Decision Tree Regression","Gradient Boosting Regression","XGBoost Regression","SVR"]
                    Classifier_model_list = ["Logistic Regressor","Ridge Classifier", "Decision Tree Classifier","Random Forest Classifier","Gradient Boosting Classifier","XGBoost Classifier","SVC"]
                    
                    col1,col2,col3= st.columns([0.3, 0.4, 0.3])
                    
                    with col1:
                        st.subheader("Features Included in Training")
                        # exp_name = col1.selectbox('Transformed Dataset', options=exp_list)

                        # select_method =col2.selectbox('Feature Selection',['Auto','Manual Selection'])
                        # if exp_name != '':
                        #     df_t = pd.read_csv(f"Experiments\{exp_name}.csv")
                            
                            # if select_method=='Auto':
                        
                        dict1 = pp.backward_elimination(scaled_df,target)
                        df_s= pd.DataFrame.from_dict(dict1,orient='index')
                        df_s.columns= ['P Value']
                        df_s['Column Name'] = dict1.keys()
                        df_s.reset_index(drop=True, inplace=True)
                        # df_s.index.names=['Column Name']
                        
                        df_data = pd.DataFrame({'Column_Name': scaled_df.columns, 'Data_Type': scaled_df.dtypes.values})
                        
                        df_with_selections = df_data.copy()
                        df_s.insert(0, "Include", False)
                        df_s['Include'] = True
                        
                        # Get dataframe row-selections from user with st.data_editor        
                        edited_df = st.data_editor(
                                                    df_s, 
                                                    hide_index=True,
                                                    column_config={"Select": st.column_config.CheckboxColumn(required=True)},
                                                    disabled=df.columns,
                                                    num_rows="dynamic"
                                                )  
                        
                        # Filter the dataframe using the temporary column, then drop the column
                        selected_rows = edited_df[edited_df.Include] 
                        if (edited_df['Include']== 1).any():
                            selected_rows.drop('Include', axis=1)
                            col_list = selected_rows['Column Name'].tolist() 

                        # col_list= list(dict1.keys())
                        
                        # if select_method=='Manual Selection':
                        #     df_data = pd.DataFrame({
                        #     'Column_Name': df_t.columns,
                        #     'Data_Type': df_t.dtypes.values
                        # })
                        #     df_with_selections = df_data.copy()
                        #     df_with_selections.insert(0, "Include", False)
                        #     df_with_selections['Include'] = df_with_selections['Column_Name'] == target
                            
                        #     # Get dataframe row-selections from user with st.data_editor
                            
                        #     edited_df = col2.data_editor(
                        #         df_with_selections, 
                        #         hide_index=True,
                        #         column_config={"Select": st.column_config.CheckboxColumn(required=True)},
                        #         disabled=df.columns,
                        #         num_rows="dynamic",
                        #     )  
                        #     # Filter the dataframe using the temporary column, then drop the column
                        #     selected_rows = edited_df[edited_df.Include] 
                        #     if (edited_df['Include']== 1).any():
                        #         selected_rows.drop('Include', axis=1)
                        #         col_list = selected_rows['Column_Name'].tolist() 
                        
                        # Specify the file path where you want to save the list
                        # file_path = "Experiments/col_list.pkl"

                        # Save the list to a file using pickle
                        with open(Path(os.path.join("Experiments", exp_name, "col_list.pkl")), 'wb') as f:
                            pickle.dump(col_list,f)
                    
                    with col2:
                        st.subheader("Models Performance")
                        
                        if ml_type=='Regression':
                            result, best_model, model_name, train_score, test_score = md.reg_model(scaled_df,col_list,target)
                            st.dataframe(result)
                            
                            with open(Path(os.path.join('Experiments', exp_name, 'best_trained_model.pkl')), 'wb') as f:
                                pickle.dump(best_model, f)
                                
                            # selected_model = col3.selectbox('Choose Model for Model Tuning',options= Regression_model_list)
                            # tuned_model, tuned_mse=md.get_hyperparameters_from_user(scaled_df,col_list,target,selected_model,col3) 
                            
                            # st.write("Tuned " + tuned_model,round(tuned_mse,4) )
                        if ml_type=='Classification':
                            result, best_model, model_name, train_score, test_score=cm.classification_model(scaled_df,col_list,target)
                            st.dataframe(result)
                            
                            with open(Path(os.path.join('Experiments', exp_name, 'best_trained_model.pkl')), 'wb') as f:
                                pickle.dump(best_model, f)
                                
                            # selected_model = col3.selectbox('Choose Model for Model Tuning',options= Classifier_model_list)
                        
                            #selected_model = col3.selectbox('Choose Model for Model Tuning',options= Classifier_model_list)
                            # tuned_model, tuned_accuracy=cm.get_hyperparameters_from_user(scaled_df,col_list,target,selected_model,col3)
                            # st.write(tuned_model, round(tuned_accuracy,4))
                    with col3:
                        st.subheader('Best Models Details')
                        st.write(f"Best Model : {model_name}")
                        st.write(f"Training Score : {train_score}")
                        st.write(f"Testing Score : {test_score}")
                        st.write(f"Model is saved at Experiments\\{exp_name}\\best_trained_model.pkl")

                    col1, col2, col3 = st.columns([0.2, 0.5, 0.3])
                    with col1:
                        tunning_flag=st.checkbox("Optimize the model performance.")
                    
                    with col2:
                        if tunning_flag:
                            st.subheader("Select Hyperparameters")
                            if ml_type=='Classification':
                                selected_model = st.selectbox('Choose Model for Model Tuning',options= ["Logistic Regressor", "DecisionTree Classifier", "RandomForest Classifier", "GradientBoosting Classifier", "XGBoost Classifier", "SVC"])
                                tuned_model, params, tuned_score_train, tuned_score_test=cm.get_hyperparameters_from_user(scaled_df,col_list,target,selected_model)
                                # st.write(tuned_model, round(tuned_score,4))
                                
                            elif ml_type=='Regression':
                                selected_model = st.selectbox('Choose Model for Model Tuning',options= Regression_model_list)
                                tuned_model, params, tuned_score_train, tuned_score_test=md.get_hyperparameters_from_user(scaled_df,col_list,target,selected_model) 
                                # st.write("Tuned " + tuned_model,round(tuned_score,4) )
                    
                            with col3:
                                st.subheader("Tuned Model's Details")
                                with open(Path(os.path.join('Experiments', exp_name, 'best_tuned_model.pkl')), 'wb') as f:
                                    pickle.dump(tuned_model, f)
                                st.write(f"Training Score : {round(tuned_score_train,4)}")
                                st.write(f"Testing Score : {round(tuned_score_test,4)}")
                                st.write(f"Model is saved at Experiments\\{exp_name}\\best_tuned_model.pkl")
                                st.write(f"Best HyperParameters:\n{params}")
                                
                                
            elif ml_type == 'Computer Vision':
                use_case_path = os.path.join(os.getcwd(), 'Experiments',exp_name, target)
                os.makedirs(use_case_path, exist_ok=True)
                # if (target == 'Object Detection through Video') | (target == 'Object Detection through Image'):  #(ml_type == 'Computer Vision') & 
                flag_col1, flag_col2, flag_col3, flag_col4 = st.columns([0.3, 0.2, 0.2, 0.3])
                with flag_col2:
                    generic_Use_case_flag = st.checkbox(label="Create Generic Use Case", value=True)
                    if generic_Use_case_flag:
                        use_case_flag='Generic'
                with flag_col3:
                    custom_Use_case_flag = st.checkbox(label="Create Custom Use Case", value=False)
                    if custom_Use_case_flag:
                        use_case_flag='Custom'
                        
                save_experiment_details(exp_name, date_time, ml_type, dataset_name, task=target, use_case_flag= use_case_flag)
                
                if generic_Use_case_flag: #(target == "Object Detection through Video") & 
                    sub_target = st.selectbox("Select Use Case", ["Generic_Object_Detection", "PPE_Kit_Detection", "Running_Detection"])
                    # os.makedirs(os.path.join('Experiment', exp_name, target, sub_target), exist_ok=True)
                    col1, col2, col3 = st.columns([0.47, 0.4, 0.2])
                    with col2:
                        create_exp_button = st.button("Create Experiment") #, on_click=click_create_experiment
                    if create_exp_button: #st.session_state.create_experiment
                        os.makedirs(os.path.join(os.getcwd(), 'Experiments', exp_name, target, sub_target), exist_ok=True)
                        st.success('Experiment creation completed')
                        
                elif custom_Use_case_flag: # (target == "Object Detection through Video") & 
                    sub_target = st.text_input('Provide Use Case Name')
                    # cwd = os.getcwd()
                    working_path = os.path.join(os.getcwd(), 'Experiments', exp_name, target, sub_target)
                    os.makedirs(working_path, exist_ok=True)
                    
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        num_classes = st.number_input("Enter number of classes:", min_value=1, value=2)
                    with col2:
                        class_names = st.text_input("Enter class names (comma-separated):")
                    with col3:
                        # Dynamically change list of YOLO models based on project typef'
                        if target.__contains__("Classification"):
                            yolo_models = ["yolov8n-cls.pt", "yolov8s-cls.pt"]
                        elif target.__contains__("Segmentation"):
                            yolo_models = ["yolov8n-seg.pt"]
                        else:
                            yolo_models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
                    
                        model_name = st.selectbox("Select YOLO model:", yolo_models)
                        
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        pretrained = st.checkbox("Use pretrained model")
                    with col2:
                        num_epochs = st.number_input("Enter number of epochs:", min_value=1, value=20)
                    with col3:
                        img_size = st.number_input("Enter image size:", min_value=32, value=416)
                    with col4:
                        batch_size = st.number_input("Enter batch size:", min_value=1, value=16)
                    
                    class_labels = class_names.split(',')
                    class_dict = {item:index for index,item in enumerate(class_labels)}
                    
                    annotation_flag = st.checkbox("Image Annotation", value=False)
                    
                    if annotation_flag:
                        
                        extracted_train_dirs = "train"   #os.path.join(working_path, "train")
                        extracted_val_dirs = "valid"   #os.path.join(working_path, "valid")
                        
                        image_files = []
                        for root, _, files in os.walk(os.path.join("Datasets", dataset_name)):
                            if not root.__contains__("__MACOSX"):
                                for file in files:
                                    if file.endswith((".jpg", ".jpeg", ".png")):
                                        image_files.append(os.path.join(root, file))
                                
                        # image_index = st.session_state.get("image_index", 0)
                        # if image_index < (len(image_files)-1):
                        #     image_path = image_files[image_index]
                        #     image = Image.open(image_path)
                        
                        st.subheader("Bounding Box Annotation")
                        st.write("Draw bounding boxes on the image below.")
                        
                        if len(image_files) == 0:
                            st.error("No image files found in the specified folder.")
                        else:
                            if 'result_dict' not in st.session_state:
                                result_dict = {}
                                for img in image_files:
                                    result_dict[img] = {'bboxes': [], 'labels': []}  # Initialize as empty lists
                                st.session_state['result_dict'] = result_dict.copy()

                            current_index = st.session_state.get('current_index', 0)

                            if st.button("Next Image") and current_index < len(image_files) - 1:
                                current_index += 1
                                st.session_state['current_index'] = current_index
                            target_image_path = image_files[current_index]

                            new_labels = detection(image_path=target_image_path, 
                                                bboxes=st.session_state['result_dict'][target_image_path]['bboxes'], 
                                                labels=st.session_state['result_dict'][target_image_path]['labels'], 
                                                label_list=class_labels,
                                                height=512,
                                                width=512,
                                                key=target_image_path)
                            
                            if new_labels is not None:
                                st.session_state['result_dict'][target_image_path]['bboxes'] = [v['bbox'] for v in new_labels]
                                st.session_state['result_dict'][target_image_path]['labels'] = [v['label_id'] for v in new_labels]

                            image_name = os.path.basename(target_image_path)
                            image = Image.open(target_image_path)
                            
                            # # Normalize box coordinates
                            # scale = max(image.size)
                            # normalized_bboxes = [[b / scale for b in bbox] for bbox in st.session_state['result_dict'][target_image_path]['bboxes']]

                            # Display the image annotations
                            # st.json(image_annotations[current_image_index])
                            
                            if (current_index + 1) % 3 == 0:  # Every third image goes to the "valid" folder
                                valid_folder = os.path.join(working_path, "valid","images")
                                labels_valid_folder = os.path.join(working_path, "valid","labels")
                                if not os.path.exists(valid_folder):
                                    os.makedirs(valid_folder)
                                if not os.path.exists(labels_valid_folder):
                                    os.makedirs(labels_valid_folder)
                                image.save(os.path.join(valid_folder, image_name))
                                
                                annotation_text_file = os.path.splitext(image_name)[0] + ".txt"
                                annotation_text_path = os.path.join(labels_valid_folder, annotation_text_file)
                                with open(annotation_text_path, "w") as annotation_file:
                                    for bbox, label_id in zip(st.session_state['result_dict'][target_image_path]['bboxes'], st.session_state['result_dict'][target_image_path]['labels']):
                                        st.write(label_id)
                                        label = label_id
                                        annotation_file.write(f"{label} {((bbox[0]+bbox[2])/2)/image.width} {((bbox[1]+bbox[3])/2)/image.height} {(bbox[2]-bbox[0])/image.width} {(bbox[3]-bbox[1])/image.height}\n")

                            else:
                                train_folder = os.path.join(working_path, "train","images")
                                labels_train_folder = os.path.join(working_path, "train","labels")
                                if not os.path.exists(train_folder):
                                    os.makedirs(train_folder)
                                if not os.path.exists(labels_train_folder):
                                    os.makedirs(labels_train_folder)
                                image.save(os.path.join(train_folder, image_name))

                                # Save the annotation text file in the "labels_train" folder
                                annotation_text_file = os.path.splitext(image_name)[0] + ".txt"
                                annotation_text_path = os.path.join(labels_train_folder, annotation_text_file)
                                with open(annotation_text_path, "w") as annotation_file:
                                    for bbox, label_id in zip(st.session_state['result_dict'][target_image_path]['bboxes'], st.session_state['result_dict'][target_image_path]['labels']):
                                        label = label_id
                                        annotation_file.write(f"{label} {((bbox[0]+bbox[2])/2)/image.width} {((bbox[1]+bbox[3])/2)/image.height} {(bbox[2]-bbox[0])/image.width} {(bbox[3]-bbox[1])/image.height}\n")
                                        
                            # Button to move to the next image
                            if current_index >= len(image_files) - 1:
                                # if st.button('Next Image'):
                                #     image_index += 1
                                #     st.session_state["image_index"] = image_index
                                st.write("End of images.")
                                training_flag = st.button("Start Model Training", key='annoated_training')
                            else:
                            #     st.write("End of images.")
                            #     training_button = st.button("Start Model Training", key='annoated_training')
                                training_flag = False
                        
                        
                        
                        
                        # Display the image
                        # col1, col2 = st.columns(2)
                        # col1.image(image, caption=f"Image {image_index + 1}/{len(image_files)}", use_column_width=True)

                        # Sidebar input for bounding box annotation
                            # st.subheader("Bounding Box Annotation")
                            # st.write("Draw bounding boxes on the image below.")

                        # # Create a canvas for drawing bounding boxes
                        # canvas = st_canvas(
                        #     fill_color="rgba(255, 165, 0, 0.1)",  # Orange color with 30% opacity
                        #     stroke_width=2,
                        #     stroke_color="rgba(255, 165, 0, 0.6)",  # Orange color with 60% opacity
                        #     background_image=image,
                        #     drawing_mode="rect",
                        #     key="canvas",
                        #     height=300,
                        #     width=300,
                        #     update_streamlit=True,
                        # )

                        # # Initialize list to store annotations
                            # annotations = []
                        # if canvas.json_data["objects"]:
                        #     st.write("Bounding Box Coordinates and Labels:")
                        #     for obj in canvas.json_data["objects"]:
                        #         if obj["type"] == "rect":
                        #             # Check if all keys are present in the obj dictionary
                        #             if all(key in obj for key in ['left', 'top', 'width', 'height']):
                                        
                        #                 box_label = st.selectbox("Label for Box", options=class_labels)
                                        
                        #                 # Normalize box coordinates
                        #                 left = obj["left"] / image.width
                        #                 top = obj["top"] / image.height
                        #                 width = obj["width"] / image.width
                        #                 height = obj["height"] / image.height
                        #                 annotations.append([class_dict[box_label], left, top, width, height])
                        #                 st.write(f"Label: {class_dict[box_label]}, Coordinates: {obj['left']}, {obj['top']}, {obj['width']}, {obj['height']}")
                        #             else:
                        #                 st.warning("Bounding box coordinates are incomplete.")
                        #         else:
                        #             st.warning("Unknown object type found in canvas data.")
                        
                        
                        
                        
                        
                        
                        
                        # new_labels = detection(image_path=image_path, 
                        #                         bboxes=st.session_state['result_dict'][image_path]['bboxes'], 
                        #                         labels=st.session_state['result_dict'][image_path]['labels'], 
                        #                         label_list=class_labels, key=image_path)
                                    
                        # col1, col2, col3 = st.columns([0.1, 0.2, 0.7])
                        # with col1:
                        #     next_img = st.button("Next Image")
                        # with col2:
                        #     training_button = st.button("Start Model Training", key='annoated_training')
                            
                        # if next_img and image_index < len(image_files):
                            
                        # #     canvas.json_data = {"objects": []}
                        # #     st.session_state["canvas_data"] = {"objects": []}
    
                        # #     # Save the annotations to a text file with the name of the image file
                        #     image_name = os.path.basename(image_path)
                        #     if (image_index + 1) % 3 == 0:  # Every third image goes to the "valid" folder
                        #         valid_folder = os.path.join(working_path, "valid","images")
                        #         labels_valid_folder = os.path.join(working_path, "valid","labels")
                        #         if not os.path.exists(valid_folder):
                        #             os.makedirs(valid_folder)
                        #         if not os.path.exists(labels_valid_folder):
                        #             os.makedirs(labels_valid_folder)
                        #         image.save(os.path.join(valid_folder, image_name))

                        #         # Save the annotation text file in the "labels_valid" folder
                        #         annotation_text_file = os.path.splitext(image_name)[0] + ".txt"
                        #         annotation_text_path = os.path.join(labels_valid_folder, annotation_text_file)
                        #         with open(annotation_text_path, "w") as annotation_file:
                        #             for annotation in annotations:
                        #                 annotation_file.write(f"{annotation[0]} {annotation[1]} {annotation[2]} {annotation[3]} {annotation[4]}\n")
                        #     else:
                        #         train_folder = os.path.join(working_path, "train","images")
                        #         labels_train_folder = os.path.join(working_path, "train","labels")
                        #         if not os.path.exists(train_folder):
                        #             os.makedirs(train_folder)
                        #         if not os.path.exists(labels_train_folder):
                        #             os.makedirs(labels_train_folder)
                        #         image.save(os.path.join(train_folder, image_name))

                        #         # Save the annotation text file in the "labels_train" folder
                        #         annotation_text_file = os.path.splitext(image_name)[0] + ".txt"
                        #         annotation_text_path = os.path.join(labels_train_folder, annotation_text_file)
                        #         with open(annotation_text_path, "w") as annotation_file:
                        #             for annotation in annotations:
                        #                 annotation_file.write(f"{annotation[0]} {annotation[1]} {annotation[2]} {annotation[3]} {annotation[4]}\n")
                                        
                                        
                        # if 'result_dict' not in st.session_state:
                        #     result_dict = {}
                        #     # for img in image_path_list:
                        #     result_dict[image_path] = {'bboxes': [[0,0,100,100],[10,20,50,150]],'labels':[0,3]}
                        #     st.session_state['result_dict'] = result_dict.copy()

                        # # num_page = st.slider('page', 0, len(image_path_list)-1, 0, key='slider')
                        # # target_image_path = image_path_list[num_page]
                        
                        # if new_labels is not None:
                        #     st.session_state['result_dict'][image_path]['bboxes'] = [v['bbox'] for v in new_labels]
                        #     st.session_state['result_dict'][image_path]['labels'] = [v['label_id'] for v in new_labels]
                            
                        # st.json(st.session_state['result_dict'])
                        #     # Move to the next image
                        #     image_index += 1
                        #     st.session_state["image_index"] = image_index
                            
                        # if end_of_img_flag:
                        #     training_button = st.button("Start Model Training", key='annoated_training')
                        #     extracted_train_dirs = "train"   #os.path.join(working_path, "train")
                        #     extracted_val_dirs = "valid"   #os.path.join(working_path, "valid")
                            
                    else:
                        extracted_train_dirs = "train"     #os.path.join(working_path, "train")
                        # if not os.path.exists(extracted_train_dirs):
                        #     os.makedirs(extracted_train_dirs)
                        
                        extracted_val_dirs = "valid"     #os.path.join(working_path, "valid")
                        # if not os.path.exists(extracted_val_dirs):
                        #     os.makedirs(extracted_val_dirs)

                        train_folder = os.path.join(working_path, "train","images")
                        labels_train_folder = os.path.join(working_path, "train","labels")
                        if not os.path.exists(train_folder):
                            os.makedirs(train_folder)
                        if not os.path.exists(labels_train_folder):
                            os.makedirs(labels_train_folder)
                        # image.save(os.path.join(train_folder, image_name))
                                
                        valid_folder = os.path.join(working_path, "valid","images")
                        labels_valid_folder = os.path.join(working_path, "valid","labels")
                        if not os.path.exists(valid_folder):
                            os.makedirs(valid_folder)
                        if not os.path.exists(labels_valid_folder):
                            os.makedirs(labels_valid_folder)
                        # image.save(os.path.join(valid_folder, image_name))
                        
                        image_files = []
                        labels_files = []
                        for files in os.listdir(os.path.join("Datasets", dataset_name)):
                            if files != "__MACOSX":
                                for file in os.listdir(os.path.join("Datasets", dataset_name, files)):
                                    if file == 'images':
                                        for image in os.listdir(os.path.join("Datasets", dataset_name, files, 'images')):
                                            image_files.append(os.path.join("Datasets", dataset_name, files, 'images', image))

                                    elif file == 'labels':
                                        for text_file in os.listdir(os.path.join("Datasets", dataset_name, files, 'labels')):
                                            labels_files.append(os.path.join("Datasets", dataset_name, files, 'labels', text_file))

                                    
                        # # Shuffle the list of files
                        # random.shuffle(image_files)

                        # Calculate the split index based on the split ratio
                        split_index = int(len(image_files) * 0.8)

                        # Split the files into train and validation sets
                        train_images = image_files[:split_index]
                        train_labels = labels_files[:split_index]
                        
                        val_images = image_files[split_index:]
                        val_labels = labels_files[split_index:]

                        # Copy train files to the train folder
                        for file in train_images:
                            shutil.copy(file, os.path.join(train_folder, os.path.split(file)[1]))
                            
                        for file in train_labels:
                            shutil.copy(file, os.path.join(labels_train_folder, os.path.split(file)[1]))

                        # Copy validation files to the validation folder
                        for file in val_images:
                            shutil.copy(file, os.path.join(valid_folder, os.path.split(file)[1]))
                        for file in val_labels:
                            shutil.copy(file, os.path.join(labels_valid_folder, os.path.split(file)[1]))
                            
                        col1, col2, col3 = st.columns([0.4, 0.3, 0.3])
                        with col2:
                            training_flag = st.button("Start Model Training", key='normal_training')
                    
                    if training_flag:
                        st.success("Model Training is in Progress...")
                        yolo_at_obj = cv.YOLO_AUTO_TRAIN(use_case_path=working_path,
                                                        val_dataset_path=extracted_val_dirs,
                                                        train_dataset_path=extracted_train_dirs,
                                                        class_labels=class_labels,
                                                        model_name=model_name,
                                                        pretrained_flag=pretrained,
                                                        num_epochs=num_epochs,
                                                        img_size=img_size,
                                                        batch_size=batch_size)
                             
                        model_save_path = yolo_at_obj.training()
                        st.success(f"Model trained successfully! Model saved at: {model_save_path}")   
                        
    if st.session_state.prediction:
        "---"
        col1, col2, col3 = st.columns([0.47, 0.4, 0.2])
        with col2:
            st.subheader('Prediction')
            
        experiments = []
        for item in os.listdir('Experiments'):
            if os.path.isdir(os.path.join('Experiments', item)):
                experiments.append(item)
                
        col1, col2, col3, col4, col5, col6 = st.columns([0.05, 0.25, 0.25, 0.25, 0.2, 0.05])
        with col2:
            exp_name = st.selectbox("Select the Experiment", options=experiments)
            
        tasks = []
        for item in os.listdir(Path(os.path.join('Experiments', exp_name))):
            if os.path.isdir(os.path.join('Experiments', exp_name, item)):
                tasks.append(item)
        with col3:
            if len(tasks) > 0:
                task = st.selectbox("Select the Task", options=tasks)
                
                sub_tasks = []
                for item in os.listdir(Path(os.path.join('Experiments', exp_name, task))):
                    if os.path.isdir(os.path.join('Experiments', exp_name, task, item)):
                        sub_tasks.append(item)
                with col4:
                    if len(sub_tasks) > 0:
                        sub_task = st.selectbox("Select the Use Cse", options=sub_tasks)
                
        _, _, ml_type, dataset_name, target, _, use_case_type = extract_experiment_details(Path(os.path.join('Experiments', exp_name, 'exp_details.txt')))
        
        if (ml_type=='Computer Vision') and (use_case_type=='Custom'):
            with col5:
                # new_data_flag = st.checkbox("Predict on New Data", value=False, key="new_data_prediction")
                test_cases = []
                for item in os.listdir(Path(os.path.join('Experiments', exp_name, task, sub_task))):
                    if os.path.isdir(os.path.join('Experiments', exp_name, task, sub_task, item)) and item.__contains__("test1_"):
                        test_cases.append(item)
                        
                test_case = st.selectbox("Select Test Case", test_cases)
            # File upload section
            custom_cv_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "mp4"])
                
        st.title("")
            
        # sub_col1, sub_col2, sub_col3 = st.columns([0.47, 0.2, 0.33])
        # with sub_col2:
        #     predict_button = st.button("Predict", key='reg_pred', on_click=click_predict)
            
        # if predict_button:
        if ml_type in['Regression', 'Classification']:
            col1,col2= st.columns(2)
            with col1:
                data_file=st.file_uploader("Upload test CSV file", type=['csv'])
            
            with col2:
                model_list = ['']
                model_list.extend([file.split('.')[0] for file in os.listdir(Path(os.path.join("Experiments", exp_name))) if file.endswith('.pkl')])
                model= st.selectbox("Select Model",model_list)
        
            if data_file and model:
                st.title("")
                input_df= pd.read_csv(data_file)
                
                with open(Path(os.path.join("Experiments", exp_name, "selected_cols.pkl")), 'rb') as f:
                    cols_list = pickle.load(f)
                    
                if not target:
                    cols_list.pop(target)
                                
                test_df = input_df[cols_list]  
                treated_df =  prediction.replace_missing_values(test_df, target)
                encoded_df, target_col= prediction.create_dummies(treated_df, target)
                
                with open(Path(os.path.join("Experiments", exp_name, "scaler.pkl")),"rb") as f:
                    scaler= pickle.load(f)
                scaled_data = scaler.transform(encoded_df)
                
                # opening the file in read mode 
                with open(Path(os.path.join("Experiments", exp_name, "col_list.pkl")), "rb") as f:
                    col_list = pickle.load(f)
                
                # Convert the scaled data back to a DataFrame (optional)
                scaled_df = pd.DataFrame(scaled_data, columns=encoded_df.columns) 
                
                scaled_df = scaled_df[col_list]
                
                with open(Path(os.path.join("Experiments", exp_name, f"{model}.pkl")), "rb") as f:
                    best_model=pickle.load(f)

                sub_col1, sub_col2, sub_col3 = st.columns([0.47, 0.2, 0.33])

                if sub_col2.button('Predict'):
                    st.subheader("")
                    y_pred = best_model.predict(scaled_df)
                    y_pred = pd.DataFrame(y_pred,columns=['Predicted Target'])

                    result_df = pd.concat([input_df, y_pred],axis=1)
                        
                    st.dataframe(result_df)
                    
        elif ml_type== 'Computer Vision':
            
            if use_case_type == 'Generic':
                if task=="Image_Segmentation":
                    background_image = None
                    use_background_image = st.checkbox("Use Background Image")
                    if use_background_image:
                        background_file = st.file_uploader("Upload Background Image", type=["jpg", "jpeg", "png"])
                        if background_file is not None:
                            background_image = Image.open(background_file)
                            
                if task.__contains__('Image'):
                    # if new_data_flag:
                    #     input_image=st.file_uploader("Upload the Image", type=["jpg", "jpeg", "png"])
                    # else:
                    input_image = Image.open(Path(os.path.join('Datasets', dataset_name)))
                        
                    if input_image:
                        col1, col2 = st.columns([0.5, 0.5])
                        with col1:
                            st.image(input_image, use_column_width=True)
                        
                        with col2:
                            if task == "Image_Classification":
                                predicted_classes = cv.classify(input_image)
                                # Display predicted classes
                                st.write("Detected Classes:")
                                for class_name, confidence in predicted_classes:
                                    st.write(f"- {class_name}: {confidence}")
                            elif task == "Object_Detection_through_Image":
                                detected_objects = cv.detect_objects(input_image)
                                # Display image with bounding boxes
                                st.image(detected_objects[0].plot()[:, :, ::-1], caption="Detected Objects", use_column_width=True)
                            elif task == "Image_Segmentation":
                                segmentation_result = cv.segment_and_draw(input_image)
                                st.image(segmentation_result, caption="Segmentation Result", use_column_width=True)
                        
                else:
                    if dataset_name != "Live Feed through WebCam":
                                
                        col1, col2 = st.columns([0.5, 0.5])
                        with col1:
                            with open(Path(os.path.join("Datasets", dataset_name, f"{dataset_name}.mp4")), 'rb') as video_file:
                                st.subheader('Original Video')
                                st.video(video_file)
                                
                        with open(Path(os.path.join("Datasets", dataset_name, f"{dataset_name}.mp4")), 'rb') as video_file:
                            with col2:
                                if sub_task == "Generic_Object_Detection":
                                    st.subheader("Object Detection Video")
                                    cv.detect_generic_objects_in_video(video_file)
                                elif sub_task == "PPE_Kit_Detection":
                                    st.subheader("PPE kit Detection Video")
                                    cv.detect_ppe_kits_in_video(video_file)
                                elif sub_task == "Running_Detection":
                                    st.subheader("Running Detection Video")
                                    cv.detect_running_in_video(video_file)
                                    
                    else:
                        # Initializing the Webcam
                        # if st.button("Start Webcam"):
                        col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
                        with col2:
                            if sub_task == "Generic_Object_Detection":
                                st.subheader("Object Detection Video")
                                cv.detect_generic_objects_in_webcam()
                            elif sub_task == "PPE_Kit_Detection":
                                st.subheader("PPE kit Detection Video")
                                cv.detect_ppe_kits_in_webcam()
                            elif sub_task == "Running_Detection":
                                st.subheader("Running Detection Video")
                                cv.detect_running_in_webcam()
                                    
            elif use_case_type == 'Custom':

                if custom_cv_file is not None:
                    
                    # Load the selected model
                    model = pred.load_cv_model(Path(os.path.join("Experiments", exp_name, task, sub_task, test_case, "weights", "model.pt")))
                    
                    col1,col2=st.columns(2)
                    if custom_cv_file.name.split(".")[-1] in ["jpg", "jpeg", "png"]:
                        # Display the uploaded image
                        image = Image.open(custom_cv_file)
                        with col1:
                            st.image(image, caption="Uploaded Image", use_column_width=True)
                            
                        with col2:
                            pred.predict_image(image, model)
                    else:
                        with col1:
                            st.video(custom_cv_file) #caption="Uploaded Video", use_column_width=True
                            
                        with col2:
                            pred.predict_video(custom_cv_file, model)       
              
if __name__ == '__main__':
    main()
            
        