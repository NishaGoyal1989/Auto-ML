import cv2
import tempfile
from ultralytics import YOLO
import os
import streamlit as st
import ultralytics
import yaml


# FOR WEBCAM
def detect_ppe_kits_in_webcam():
    # Load your YOLOv5 model
    model= YOLO("Models/best.pt")
    
    # Read the video from the provided source
    vid_cap = cv2.VideoCapture(0)
    st_frame = st.empty()
    
    while (vid_cap.isOpened()):
        success, image = vid_cap.read()
        if success:
            # Resize frame if necessary
            image = cv2.resize(image, (720, int(720*(9/16))))
            
            # Perform detection
            res = model.predict(image)
            result_tensor = res[0].boxes
            res_plotted = res[0].plot()
            
            # Display detected objects
            st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)
        else:
            vid_cap.release()
            break

def detect_generic_objects_in_webcam():
    # Load your YOLOv5 model
    model = YOLO("Models/yolov8n.pt")
    
    # Read the video from the provided source
    vid_cap = cv2.VideoCapture(0)
    st_frame = st.empty()
    
    while (vid_cap.isOpened()):
        success, image = vid_cap.read()
        if success:
            # Resize frame if necessary
            image = cv2.resize(image, (720, int(720*(9/16))))
            
            # Perform detection
            res = model.predict(image)
            result_tensor = res[0].boxes
            res_plotted = res[0].plot()
            
            # Display detected objects
            st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)
        else:
            vid_cap.release()
            break
# FOR VIDEO-SOURCE
def detect_ppe_kits_in_video(uploaded_video):
    # Load your YOLOv5 model
    #model = YOLO('yolov8n.pt')
    model= YOLO('Models/best.pt')
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
            image = cv2.resize(image, (720, int(720*(9/16))))
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
def detect_generic_objects_in_video(uploaded_video):
    # Load your YOLOv5 model
    #model = YOLO('yolov8n.pt')
    model= YOLO('Models/yolov8n.pt')
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
            image = cv2.resize(image, (720, int(720*(9/16))))
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
        
def detect_ppe_kits_in_video(uploaded_video):
    # Load your YOLOv5 model
    #model = YOLO('yolov8n.pt')
    model= YOLO('Models/best.pt')
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
            image = cv2.resize(image, (720, int(720*(9/16))))
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
    model= YOLO('Models/Running_new.pt')
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
            image = cv2.resize(image, (720, int(720*(9/16))))
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
    model = YOLO("Models/Running_new.pt")
    
    # Read the video from the provided source
    vid_cap = cv2.VideoCapture(0)
    st_frame = st.empty()
    
    while (vid_cap.isOpened()):
        success, image = vid_cap.read()
        if success:
            # Resize frame if necessary
            image = cv2.resize(image, (720, int(720*(9/16))))
            
            # Perform detection
            res = model.predict(image)
            
            res_plotted = res[0].plot()
            
            # Display detected objects
            st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)
        else:
            vid_cap.release()
            break

class YOLO_AUTO_TRAIN:
    def __init__(self):
        pass
    
    def Create_yaml(self, train_dataset_path, val_dataset_path, num_class, class_names, user_name, project_name):
        
        #train_path = os.path.join(user_name, project_name, train_dataset_path)
        #val_path = os.path.join(user_name, project_name, val_dataset_path)
        yaml_string = dict(train=train_dataset_path, val=val_dataset_path, nc=num_class, names=class_names)
        with open(os.path.join(user_name, project_name, "data.yaml"), "w") as yaml_file:
            yaml.dump(yaml_string, yaml_file, default_flow_style=False, sort_keys=False)
            
    def Create_folder(self,user_name,project_name):
        if not os.path.exists(user_name):
            os.makedirs(user_name)
        if not os.path.exists(user_name+"/"+project_name):
            os.makedirs(user_name+"/"+project_name)
        
    def training(self,user_name ,val_dataset_path ,train_dataset_path ,class_names ,num_class ,model_name="yolov8m.pt" ,pretrained=False ,project_name="YOLO_V8" ,experiment_name="test1_" ,save_model_name='model.pt' ,num_epochs=2 ,imgsz=640 ,batch_size=-1 ,device="cpu"):
            print("=============== Start Training ================")
            user_name=user_name
            self.Create_folder(user_name,project_name)
            data=self.Create_yaml(train_dataset_path,val_dataset_path,num_class,class_names,user_name,project_name)
            #data="data.yaml"
            model = ultralytics.YOLO(model_name)

            model.train(data=user_name+"/"+project_name+"/data.yaml",
                                epochs=num_epochs,
                                batch=batch_size,
                                device=device,
                                project=user_name+"/"+project_name,
                                name=experiment_name,
                                pretrained=pretrained, 
                                imgsz=imgsz,
                               )

            #Export Model returns path where model is saved
            model_save_path =model.export()
            model_save_path=model_save_path.split(".")
            model_save_path[-1]=".pt"
            model_save_path="".join(model_save_path)
            new_model_save_path=model_save_path.replace("best.pt",str(save_model_name)+".pt")
            #------------------#
            os.rename(model_save_path,new_model_save_path)
            return new_model_save_path
   #cap = cv2.VideoCapture(video_path)

#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#     #frame_rate = int(cap.get(5))

#     # Define a VideoWriter to create the processed video
#     fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
#    # Define a temporary directory for processed video
#     #temp_output_dir = tempfile.TemporaryDirectory()
#     output_path = os.path.join('video', "out.mp4")
#     out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_height, frame_width),True)
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # Perform object detection on the frame
#         results = model.track(frame,persist =True)
#         frame_=results[0].plot()
#         frame_ = cv2.cvtColor(frame_, cv2.COLOR_BGRA2BGR)
#         #st.image(frame_)
#         out.write(frame_) 
         
#         #st.write("Frame is done")     
#     cap.release()
#     out.release()
#     # Closes all the frames 
#     cv2.destroyAllWindows() 
#     return output_path