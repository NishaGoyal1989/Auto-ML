import pandas as pd
import streamlit as st
from PIL import Image
import ultralytics
import tempfile
import cv2

@st.cache_data
def replace_missing_values(df, target):
    
    if target:
        df_target=df[target]
        df=df.drop([target], axis=1)
        
    # Separate columns by data types
    numeric_dtype_cols = df.select_dtypes(exclude=['object']).columns.tolist()
    categorical_dtype_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Replace missing values with mean for numerical columns
    for col in numeric_dtype_cols:
        if df[col].isnull().sum() > 0:  # Check if the column has missing values
            mean_val = df[col].mean()  # Calculate mean value for the column
            df[col].fillna(mean_val, inplace=True)  # Replace missing values with mean

    # Replace missing values with median for categorical columns
    for col in categorical_dtype_cols:
        if df[col].isnull().sum() > 0:  # Check if the column has missing values
            mfv_val = df[col].mode()  # Calculate median value for the column
            df[col].fillna(mfv_val, inplace=True)  # Replace missing values with median
       
    if target:     
        df = pd.concat([df,df_target], axis=1)
        return df
    else:
        return df

# Function to create dummy variables for categorical columns
@st.cache_data
def create_dummies(df, target):
    if target:
        df_target=df[target]
        df=df.drop(target, axis=1)
        
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Create dummy variables for each categorical column
    df_dummies = pd.get_dummies(df, columns=categorical_cols,dtype =int)
 
    return df_dummies, df_target


# Function to load the selected model
def load_cv_model(model_path):
    # Load the model using torch.load or your preferred method
    st.write(model_path)
    model= ultralytics.YOLO(model_path)
    return model

# Function to make predictions
def predict_image(image, model):
    results = model.predict(image)
    st.image(results[0].plot()[:, :, ::-1], caption="Detected Objects", use_column_width=True)

def predict_video(uploaded_video,model):
    # Load your YOLOv5 model
    #model = YOLO('yolov8n.pt')
    
    # Save the uploaded video to a temporary file
    video_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_temp_file.write(uploaded_video.read())
    uploaded_video.close()
    video_temp_file.close()

    # Read the uploaded video from the temporary file
    video_path = video_temp_file.name
    vid_cap = cv2.VideoCapture(
            video_path)
    st_frame = st.empty()
    while (vid_cap.isOpened()):
        success, image = vid_cap.read()
        if success:
            height, width = image.shape[:2]
            new_width = 720
            new_height = int(height * (new_width / width))
            image = cv2.resize(image, (new_width, new_height))
            res = model.predict(image)
            res_plotted = res[0].plot()
            st_frame.image(res_plotted,
                            caption='Detected Video',
                            channels="BGR",
                            use_column_width=True
                            )
        else:
            vid_cap.release()
            break