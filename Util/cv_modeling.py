import streamlit as st
import os
from PIL import Image
import ultralytics
from ultralytics import YOLO
from pathlib import Path
import requests
from io import BytesIO
import cv2
import numpy as np
import subprocess
import tempfile
import zipfile
import yaml
from datetime import datetime as dt

# Load YOLO models
classification_model = YOLO('yolov8n-cls.pt')
detection_model = YOLO('yolov8n.pt')
segmentation_model = YOLO('yolov8x-seg.pt')

# Function to extract images from a ZIP file
@st.cache_data
def extract_images_from_zip(zip_file, extract_dir, folder_name):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    extract_dir = os.path.join(extract_dir, folder_name)

    extracted_files = [os.path.join(extract_dir, file) for file in os.listdir(extract_dir) if file.endswith((".jpg", ".jpeg", ".png"))]
    # st.write(extract_dir)
    return extracted_files

# Function to extract ZIP file and return the extracted directory's full path
@st.cache_data
def extract_zip(zip_file, extract_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    extracted_dirs = [os.path.join(extract_dir, d) for d in os.listdir(extract_dir) if os.path.isdir(os.path.normpath(os.path.join(extract_dir, d)))]
    return extracted_dirs

# Function to perform segmentation and draw filled polygons on the image
@st.cache_data
def segment_and_draw(image,background_image=None):
    
    # Run segmentation model
    results = segmentation_model.predict(image)
    
    # Convert the PIL image to OpenCV format
    img = np.array(image)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Create an empty canvas to draw the masks
    mask = np.zeros_like(img)
    # Generate random colors for each segment
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(len(results[0].masks))]

    
    
    for result in results:
        # Loop through the masks and boxes   
        for mask_c, _ in zip(result.masks.xy, result.boxes):        
            # Convert the mask coordinates to numpy array
            points = np.int32([mask_c])         
            cv2.fillPoly(mask, [points],(255,255,255))
    #segmented_image = cv2.bitwise_and(img, mask)   
    # Invert the mask to get the segmented objects
    inverted_mask = cv2.bitwise_not(mask)  
    
    # Apply the original mask to the background image to overlay the segmented objects
    segmented_image = cv2.bitwise_and(img, mask)
    
    # bimg= np.array(background_image)
    # # Apply the inverted mask to the background image to remove the segmented region
    # background_without_segment = cv2.bitwise_and(bimg, inverted_mask)
    # # Combine the segmented objects with the background image
    # alpha=0.45
    # final_image = cv2.addWeighted(segmented_image, alpha, background_without_segment, 1 - alpha, 0)
     
    return segmented_image
   
# Function to run classification predictions
@st.cache_data    
def classify(image):
    results = classification_model.predict(image)
    list1 = results[0].probs.top5
    list2 = results[0].probs.top5conf.tolist()
    names_dict = results[0].names
    names = [(names_dict[item], round(confidence, 2)) for item, confidence in zip(list1, list2)]
    return names

# Function to run object detection predictions
# @st.cache_data
def detect_objects(image):
    results = detection_model.predict(image)
    return results

# Function to run image segmentation predictions
# @st.cache_data
def segment_image(image):
    results = segmentation_model.predict(image)
    return results

# FOR WEBCAM
# @st.cache_data
def detect_ppe_kits_in_webcam():
    # Load your YOLOv5 model
    model= YOLO(os.path.join("Models", "best.pt"))
    
    # Read the video from the provided source
    vid_cap = cv2.VideoCapture(0)
    st_frame = st.empty()
    
    while (vid_cap.isOpened()):
        success, image = vid_cap.read()
        if success:
            height, width = image.shape[:2]
            new_width = 720
            new_height = int(height * (new_width / width))
            image = cv2.resize(image, (new_width, new_height))
            
            # Perform detection
            res = model.predict(image)
            result_tensor = res[0].boxes
            res_plotted = res[0].plot()
            
            # Display detected objects
            st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)
        else:
            vid_cap.release()
            cv2.destroyAllWindows()
            break

# @st.cache_data
def detect_generic_objects_in_webcam():
    # Load your YOLOv5 model
    model = YOLO(os.path.join("Models", "yolov8n.pt"))
    
    st.write("IN WEBCAM")
    # Read the video from the provided source
    vid_cap = cv2.VideoCapture(0)
    st_frame = st.empty()
    
    while (vid_cap.isOpened()):
        success, image = vid_cap.read()
        if success:
            height, width = image.shape[:2]
            new_width = 720
            new_height = int(height * (new_width / width))
            image = cv2.resize(image, (new_width, new_height))
            
            # Perform detection
            res = model.predict(image)
            result_tensor = res[0].boxes
            res_plotted = res[0].plot()
            
            # Display detected objects
            st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)
        else:
            vid_cap.release()
            cv2.destroyAllWindows()
            break
        
# FOR VIDEO-SOURCE
# @st.cache_data
def detect_ppe_kits_in_video(uploaded_video):
    # Load your YOLOv5 model
    #model = YOLO('yolov8n.pt')
    model= YOLO(os.path.join('Models', 'best.pt'))
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
            result_tensor = res[0].boxes
            res_plotted = res[0].plot()
            st_frame.image(res_plotted,
                            caption='Detected Video',
                            channels="BGR",
                            use_column_width=True
                            )
        else:
            vid_cap.release()
            break

# @st.cache_data    
def detect_generic_objects_in_video(uploaded_video):
    # Load your YOLOv5 model
    #model = YOLO('yolov8n.pt')
    model= YOLO(os.path.join('Models', 'yolov8n.pt'))
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
            result_tensor = res[0].boxes
            res_plotted = res[0].plot()
            st_frame.image(res_plotted,
                            caption='Detected Video',
                            channels="BGR",
                            use_column_width=True
                            )
        else:
            vid_cap.release()
            break

# @st.cache_data       
def detect_ppe_kits_in_video(uploaded_video):
    # Load your YOLOv5 model
    #model = YOLO('yolov8n.pt')
    model= YOLO(os.path.join('Models', 'best.pt'))
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
            result_tensor = res[0].boxes
            res_plotted = res[0].plot()
            st_frame.image(res_plotted,
                            caption='Detected Video',
                            channels="BGR",
                            use_column_width=True
                            )
        else:
            vid_cap.release()
            break
        
def detect_running_in_video(uploaded_video):
    # Load your YOLOv5 model
    #model = YOLO('yolov8n.pt')
    model= YOLO(os.path.join('Models', 'Running_new.pt'))
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
            result_tensor = res[0].boxes
            res_plotted = res[0].plot()
            st_frame.image(res_plotted,
                            caption='Detected Video',
                            channels="BGR",
                            use_column_width=True
                            )
        else:
            vid_cap.release()
            break
        
def detect_running_in_webcam():
    # Load your YOLOv5 model
    model = YOLO(os.path.join("Models", "Running_new.pt"))
    
    # Read the video from the provided source
    vid_cap = cv2.VideoCapture(0)
    st_frame = st.empty()
    
    while (vid_cap.isOpened()):
        success, image = vid_cap.read()
        if success:
            height, width = image.shape[:2]
            new_width = 720
            new_height = int(height * (new_width / width))
            image = cv2.resize(image, (new_width, new_height))
            
            # Perform detection
            res = model.predict(image)
            
            res_plotted = res[0].plot()
            
            # Display detected objects
            st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)
        else:
            vid_cap.release()
            cv2.destroyAllWindows()
            break
        
class YOLO_AUTO_TRAIN():
    def __init__(self,
                 use_case_path,
                 val_dataset_path,
                 train_dataset_path,
                 class_labels,
                 model_name="yolov8m.pt",
                 experiment_name="test1_",
                 pretrained_flag=False,
                 num_epochs=2,
                 img_size=640,
                 batch_size=-1,
                 save_model_name='model',
                 device="cpu"):
        
        self.use_case_path=use_case_path
        self.val_dataset_path=val_dataset_path
        self.train_dataset_path=train_dataset_path
        self.class_labels=class_labels
        self.model_name=model_name
        self.experiment_name=experiment_name
        self.pretrained_flag=pretrained_flag
        self.num_epochs=num_epochs
        self.img_size=img_size
        self.batch_size=batch_size
        self.device=device
        self.save_model_name = save_model_name
    
    def Create_yaml(self):
        
        yaml_string = dict(train=self.train_dataset_path,val=self.val_dataset_path, nc=len(self.class_labels), names=self.class_labels)
        with open(Path(os.path.join(self.use_case_path, "data.yaml")), "w") as yaml_file:
            yaml.dump(yaml_string, yaml_file, default_flow_style=False, sort_keys=False)        
     
    def training(self):
            print("=============== Start Training ================")
            data=self.Create_yaml()
            model = ultralytics.YOLO(self.model_name)
            
            data_path = os.path.join(self.use_case_path, "data.yaml")
            # data_url = data_path.replace('/', '\\')

            model.train(data=data_path, epochs=self.num_epochs, batch=self.batch_size, device=self.device, project=self.use_case_path, name=self.experiment_name, pretrained=self.pretrained_flag, imgsz=self.img_size)

            #Export Model returns path where model is saved
            model_save_path =model.export()
            model_save_path=model_save_path.split(".")
            model_save_path[-1]=".pt"
            model_save_path="".join(model_save_path)
            new_model_save_path=model_save_path.replace("best.pt",str(self.save_model_name)+".pt")
            #------------------#
            os.rename(model_save_path,new_model_save_path)
            return new_model_save_path
        
        
# # Streamlit UI
# option = st.sidebar.selectbox("Choose an option:", ("Image Classification", "Object Detection", "Image Segmentation","Video Object Detection"))

# if option=="Image Segmentation":
#     background_image = None
#     use_background_image = st.sidebar.checkbox("Use Background Image")
#     if use_background_image:
#         background_file = st.sidebar.file_uploader("Upload Background Image", type=["jpg", "jpeg", "png"])
#         if background_file is not None:
#                             # Open and display the uploaded background image
#             background_image = Image.open(background_file)
#             #st.image(background_image, caption="Background Image", use_column_width=True)
# if option == "Image Classification" or option == "Object Detection" or option == "Image Segmentation":
#     image_option = st.sidebar.radio("Select image input option:", ("Upload an image", "Provide image URL"))
#     if image_option == "Upload an image":
#         uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#         col1,col2= st.columns(2)
#         if uploaded_file is not None:
#             try:
#                 # Open and display the uploaded image
#                 image = Image.open(uploaded_file)
#                 col1.image(image, caption="Uploaded Image", use_column_width=True)
                
#                 # Run predictions based on selected option
#                 if option == "Image Classification":
#                     predicted_classes = classify(image)
                    
#                     # Display predicted classes
#                     col2.write("Detected Classes:")
#                     for class_name, confidence in predicted_classes:
#                         col2.write(f"- {class_name}: {confidence}")
#                 elif option == "Object Detection":
#                     detected_objects = detect_objects(image)
                    
#                     # Display image with bounding boxes
#                     col2.image(detected_objects[0].plot()[:, :, ::-1], caption="Detected Objects", use_column_width=True)
#                 elif option == "Image Segmentation":
#                     segmentation_result = segment_and_draw(image,background_image)
#                     col2.image(segmentation_result, caption="Segmentation Result", use_column_width=True)
#             except Exception as e:
#                 st.error(f"Error: {e}")

#     elif image_option == "Provide image URL":
#         image_url = st.text_input("Enter the image URL:")
#         if st.button("Fetch Image"):
#             col1,col2= st.columns(2)
#             try:
#                 # Download the image from the URL
#                 response = requests.get(image_url)
#                 image = Image.open(BytesIO(response.content))
#                 col1.image(image, caption="Image from URL", use_column_width=True)
                
#                 # Run predictions based on selected option
#                 if option == "Image Classification":
#                     predicted_classes = classify(image)
                    
#                     # Display predicted classes
#                     col2.write("Detected Classes:")
#                     for class_name, confidence in predicted_classes:
#                         col2.write(f"- {class_name}: {confidence}")
#                 elif option =="Object Detection":
#                     detected_objects = detect_objects(image)
                    
#                     # Display image with bounding boxes
#                     col2.image(detected_objects[0].plot()[:, :, ::-1], caption="Detected Objects", use_column_width=True)
#                 elif option == "Image Segmentation":
#                     segmentation_result = segment_and_draw(image,background_image)
#                     col2.image(segmentation_result, caption="Segmentation Result", use_column_width=True) 
#             except Exception as e:
#                 st.error(f"Error: {e}")
# elif option == "Video Object Detection":
#     uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
#     col1,col2=st.columns(2)
#     if uploaded_video is not None:
#         col1.video(uploaded_video)
        
#         if st.button("Detect Objects"):
#             col1.write("Processing video...")
#             utils.detect_objects_in_video(uploaded_video)

            # converted_video = "test.mp4"
            # subprocess.call(f"ffmpeg -y -i {result_video} -c:v libx264 {converted_video}", shell=True)

            # # Display the processed video
            # video_file = open(converted_video, 'rb')
            
            # col2.video(video_file.read())
            # col2.write('<div style="text-align: center; font-weight: bold;">Processed Video.</div>', unsafe_allow_html=True)