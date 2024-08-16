import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import requests
import io
from io import BytesIO
import cv2
import os
from Util.cv_modeling import extract_zip


###################################################################################################################
# Create Dataset from uploaded file
@st.cache_data
def make_dataset(data_source, input_data, dataset_name, data_type):
    
    if data_source == "Upload file from your computer":
        
        os.makedirs(os.path.join("Datasets", dataset_name), exist_ok=True)
        # data_file = st.file_uploader("Upload data file from your computer", type=None)
        # if input_data:
        if data_type == 'Tabular':
            global df
            try:
                df = pd.read_csv(input_data)
            except:
                df = pd.read_excel(input_data)
            
            # Storing the Data as csv   
            df.to_csv(os.path.join("Datasets", dataset_name, f"{dataset_name}.csv"), index=False)
            return fetch_dataframe(dataset_name)
        
        elif data_type == 'Image':
            if input_data.name.split('.')[-1] == 'zip':
                # Extract uploaded ZIP files and get full paths
                extract_zip(input_data, os.path.join("Datasets", dataset_name))
                return True
            
            elif input_data.name.split('.')[-1] in ['jpg', 'png', 'jpeg']:
                try:
                    input_image = Image.open(input_data)
                    input_image = input_image.convert("RGB") 
                    input_image.save(os.path.join("Datasets", dataset_name, f"{dataset_name}.jpg"))
                    return fetch_image(dataset_name)
                except:
                    raise Exception("Error while reading the image data")
        
        elif data_type == 'Video':
            # vid = data_file.name
            with open(os.path.join("Datasets", dataset_name, f"{dataset_name}.mp4"), mode='wb') as f:
                f.write(input_data.read())
            return fetch_video(dataset_name)
        
        else:
            st.error("Development under progress for this DataType")

    elif data_source == "Upload through url":
        
        # url = st.text_input("Enter the image URL:")

        # if url:
        if data_type == 'Tabular':
            responce = requests.get(input_data).content
            df = pd.read_csv(io.StringIO(responce.decode('utf-8')))
            df.to_csv(os.path.join("Datasets", dataset_name, f"{dataset_name}.csv"), index=False)
            return fetch_dataframe(dataset_name)
        
        elif data_type == 'Image':
            response = requests.get(input_data)
            input_image = Image.open(BytesIO(response.content))
            input_image = input_image.convert("RGB")
            input_image.save(os.path.join("Datasets", dataset_name, f"{dataset_name}.jpg"))
            return fetch_image(dataset_name)
        
        # elif data_type == 'Video':
            # cap = cv2.VideoCapture(0)

            # # Define the codec and create VideoWriter object
            # #fourcc = cv2.cv.CV_FOURCC(*'DIVX')
            # #out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
            # out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

            # while(cap.isOpened()):
            #     ret, frame = cap.read()
            #     if ret==True:
            #         frame = cv2.flip(frame,0)

            #         # write the flipped frame
            #         out.write(frame)

            #         cv2.imshow('frame',frame)
            #         if cv2.waitKey(1) & 0xFF == ord('q'):
            #             break
            #     else:
            #         break

            # # Release everything if job is finished
            # cap.release()
            # out.release()
            # cv2.destroyAllWindows()
            
        else:
            st.error("Development under progress for this DataType")
                    
    else:
        st.error("Development under progress for this DataType")
        
        # elif data_type == 'Image':
        #     background_file = st.sidebar.file_uploader("Upload Background Image", type=["jpg", "jpeg", "png"])
#######################################################################################################

######################################################################################################
# @st.cache_data
def fetch_dataframe(dataset_name):
    df = pd.read_csv(os.path.join("Datasets", dataset_name, f"{dataset_name}.csv"))
    return df
######################################################################################################

######################################################################################################
# @st.cache_data
def fetch_image(dataset_name):
    image = Image.open(os.path.join("Datasets", dataset_name, f"{dataset_name}.jpg"))
    return image
######################################################################################################

######################################################################################################
# @st.cache_data
def fetch_video(dataset_name):
    with open(os.path.join("Datasets", dataset_name, f"{dataset_name}.mp4"), mode='rb') as f:
        video = f.read()
    return video
######################################################################################################

#######################################################################################################
# Function to Check uploaded data
# @st.cache_data
def upload_checker(dataset_name=None, data_type=None, input_data=None):
    if not dataset_name:
            return "Pass the Dataset Name"
    elif not data_type:
        return "Provide the Type of Data in Dataset"
    elif not input_data:
            return "Upload data to create a dataset"  
    else:
        return False

######################################################################################################

#######################################################################################################
# Dataset Preview
@st.cache_data
def get_dataset_preview(df):
    # if upload_checker():
    #     error = upload_checker()
    #     st.write(f":red[{error}]")          
    # else:               
    # Dataset Summary
    total_cols = len(df.columns)
    total_rows = df.shape[0]
    num_cols = len(df._get_numeric_data().columns)   
    catg_cols = total_cols - num_cols
    
    st.markdown("**Dataset Summary**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"Total Columns :    {total_cols}")
        st.markdown(f"Total Rows :       {total_rows}")
    with col2:
        st.markdown(f"Numeric Columns :    {num_cols}({np.round(num_cols/total_cols * 100,2)}%)")
        st.markdown(f"Categorical Columns :       {catg_cols}({np.round(catg_cols/total_cols * 100,2)}%)")
    
    st.dataframe(df.head(50))
###############################################################################################

