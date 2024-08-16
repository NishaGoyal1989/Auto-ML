import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import requests
from io import BytesIO
import cv2
import numpy as np
import subprocess 
import Util.cv_utils as cv_utils 
import AutoML_Services.annotate_cv as annotate_cv

# Set page configuration
st.set_page_config(page_title='YOLOv8 Image and Video analysis', page_icon="Images/company_logo.jpeg", layout='wide')

# Title and sidebar
st.sidebar.image("Images/company_logo.jpeg", use_column_width=True)
st.title("YOLOv8 Image Classification, Object Detection and Image Segmenatation")

# Load YOLO models
classification_model = YOLO('yolov8n-cls.pt')
detection_model = YOLO('yolov8n.pt')
segmentation_model = YOLO('yolov8x-seg.pt')
# Function to perform segmentation and draw filled polygons on the image
def segment_and_draw(image,background_image):
    
    # Run segmentation model
    results = segmentation_model.predict(image)
    
    # Convert the PIL image to OpenCV format
    img = np.array(image)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Create an empty canvas to draw the masks
    mask = np.zeros_like(img)
    # Generate random colors for each segment
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(len(results[0].masks))]

    bimg= np.array(background_image)
    
    for result in results:
        # Loop through the masks and boxes   
        for mask_c, _ in zip(result.masks.xy, result.boxes):        
            # Convert the mask coordinates to numpy array
            points = np.int32([mask_c])         
            cv2.fillPoly(mask, [points],(255,255,255))
    #segmented_image = cv2.bitwise_and(img, mask)   
    # Invert the mask to get the segmented objects
    inverted_mask = cv2.bitwise_not(mask)         
    # Apply the inverted mask to the background image to remove the segmented region
    background_without_segment = cv2.bitwise_and(bimg, inverted_mask)
    
    # Apply the original mask to the background image to overlay the segmented objects
    segmented_image = cv2.bitwise_and(img, mask)
    
    # Combine the segmented objects with the background image
    alpha=0.45
    final_image = cv2.addWeighted(segmented_image, alpha, background_without_segment, 1 - alpha, 0)
     
    return final_image
   
# Function to run classification predictions    
def classify(image):
    results = classification_model.predict(image)
    list1 = results[0].probs.top5
    list2 = results[0].probs.top5conf.tolist()
    names_dict = results[0].names
    names = [(names_dict[item], round(confidence, 2)) for item, confidence in zip(list1, list2)]
    return names

# Function to run object detection predictions
def detect_objects(image):
    results = detection_model.predict(image)
    return results

# Function to run image segmentation predictions
def segment_image(image):
    results = segmentation_model.predict(image)
    return results

# Streamlit UI
option = st.sidebar.selectbox("Choose an option:", ("Image Classification", "Object Detection", "Image Segmentation","Video Object Detection","Add new use case"))

if option=="Image Segmentation":
    background_image = None
    use_background_image = st.sidebar.checkbox("Use Background Image")
    if use_background_image:
        background_file = st.sidebar.file_uploader("Upload Background Image", type=["jpg", "jpeg", "png"])
        if background_file is not None:
                            # Open and display the uploaded background image
            background_image = Image.open(background_file)
            #st.image(background_image, caption="Background Image", use_column_width=True)
if option == "Image Classification" or option == "Object Detection" or option == "Image Segmentation":
    image_option = st.sidebar.radio("Select image input option:", ("Upload an image", "Provide image URL"))
    if image_option == "Upload an image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        col1,col2= st.columns(2)
        if uploaded_file is not None:
            try:
                # Open and display the uploaded image
                image = Image.open(uploaded_file)
                col1.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Run predictions based on selected option
                if option == "Image Classification":
                    predicted_classes = classify(image)
                    
                    # Display predicted classes
                    col2.write("Detected Classes:")
                    for class_name, confidence in predicted_classes:
                        col2.write(f"- {class_name}: {confidence}")
                elif option == "Object Detection":
                    detected_objects = detect_objects(image)
                    
                    # Display image with bounding boxes
                    col2.image(detected_objects[0].plot()[:, :, ::-1], caption="Detected Objects", use_column_width=True)
                elif option == "Image Segmentation":
                    segmentation_result = segment_and_draw(image,background_image)
                    col2.image(segmentation_result, caption="Segmentation Result", use_column_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    elif image_option == "Provide image URL":
        image_url = st.text_input("Enter the image URL:")
        if st.button("Fetch Image"):
            col1,col2= st.columns(2)
            try:
                # Download the image from the URL
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))
                col1.image(image, caption="Image from URL", use_column_width=True)
                
                # Run predictions based on selected option
                if option == "Image Classification":
                    predicted_classes = classify(image)
                    
                    # Display predicted classes
                    col2.write("Detected Classes:")
                    for class_name, confidence in predicted_classes:
                        col2.write(f"- {class_name}: {confidence}")
                elif option =="Object Detection":
                    detected_objects = detect_objects(image)
                    
                    # Display image with bounding boxes
                    col2.image(detected_objects[0].plot()[:, :, ::-1], caption="Detected Objects", use_column_width=True)
                elif option == "Image Segmentation":
                    segmentation_result = segment_and_draw(image,background_image)
                    col2.image(segmentation_result, caption="Segmentation Result", use_column_width=True) 
               
            except Exception as e:
                st.error(f"Error: {e}")
elif option == "Video Object Detection":
    st.header("Video Object Detection")
    video_source = st.radio("Select a Video Source:", ("Upload a video", "Use Webcam"))
    if video_source == "Upload a video":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
        col1,col2=st.columns(2)
        if uploaded_video is not None:
           col1.video(uploaded_video)
           col1, col2 = st.columns(2)
  
        
           # Options for different types of detection
           detection_option = st.selectbox("Select type of detection:", ("Generic Object Detection", "PPE Kit Detection", "Running Detection"))
        
           if st.button("Detect Objects"):
              if detection_option == "Generic Object Detection":
                col1.write("Processing video for generic object detection...")
                cv_utils.detect_generic_objects_in_video(uploaded_video)
              elif detection_option == "PPE Kit Detection":
                col1.write("Processing video for PPE kit detection...")
                cv_utils.detect_ppe_kits_in_video(uploaded_video)
              elif detection_option == "Running Detection":
                col1.write("Processing video for running detection...")
                cv_utils.detect_running_in_video(uploaded_video)
              
                  
    
    # Webcam option
    elif video_source == "Use Webcam":
        st.header("Webcam Object Detection")
        detection_option = st.selectbox("Select type of detection:", ("Generic Object Detection", "PPE Kit Detection", "Running Detection"))

        # Initializing the Webcam
        if st.button("Start Webcam"):
            video_capture = cv2.VideoCapture(0)
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                if detection_option == "Generic Object Detection":
                    st.write("Processing video for generic object detection...")
                    cv_utils.detect_generic_objects_in_webcam()
                    
                elif detection_option == "PPE Kit Detection":
                    st.write("Processing video for PPE kit detection...")
                    cv_utils.detect_ppe_kits_in_webcam()
                elif detection_option == "Running Detection":
                    st.write("Processing video for running detection...")
                    cv_utils.detect_running_in_webcam()
                
                # Displaying the frames
                st.image(frame, channels="BGR")
                if cv2.waitKey(1) & 0xFF ==ord('q'):
                    break
            
            video_capture.release()
            cv2.destroyAllWindows()
        
elif option=="Add new use case":
    annotate_cv.main()         
        # if st.button("Detect Objects"):
        #     col1.write("Processing video...")
        #     utils.detect_objects_in_video(uploaded_video)

            # converted_video = "test.mp4"
            # subprocess.call(f"ffmpeg -y -i {result_video} -c:v libx264 {converted_video}", shell=True)

            # # Display the processed video
            # video_file = open(converted_video, 'rb')
            
            # col2.video(video_file.read())
            # col2.write('<div style="text-align: center; font-weight: bold;">Processed Video.</div>', unsafe_allow_html=True)