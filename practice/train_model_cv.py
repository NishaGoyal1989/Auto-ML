import streamlit as st
from Util.cv_utils import YOLO_AUTO_TRAIN  # Assuming your YOLO_AUTO_TRAIN class is defined in yolo_auto_train.py
import streamlit as st
import os
import zipfile


# Function to extract ZIP file and return the extracted directory's full path
def extract_zip(zip_file, extract_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    extracted_dirs = [os.path.join(extract_dir, d) for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
    return extracted_dirs

# Define the Streamlit app
def main():
    st.title("YOLO Model Training App")

    st.sidebar.header("User Inputs")
    username = st.sidebar.text_input("Enter username:")
    project_name = st.sidebar.text_input("Enter project name:")
    project_type = st.sidebar.selectbox("Select project type:", ["Classification", "Segmentation", "Object Detection"])
    class_names = st.sidebar.text_input("Enter class names (comma-separated):")
    num_classes = st.sidebar.number_input("Enter number of classes:", min_value=1, value=2)
    # Dynamically change list of YOLO models based on project type
    if project_type == "Classification":
        yolo_models = ["yolov8n-cls.pt", "yolov8s-cls.pt"]
    elif project_type == "Segmentation":
        yolo_models = ["yolov8n-seg.pt"]
    elif project_type == "Object Detection":
        yolo_models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
    else:
        st.error("Invalid project type selected.")
        return
    
    model_name = st.sidebar.selectbox("Select YOLO model:", yolo_models)
    #model_name = st.sidebar.selectbox("Select YOLO model:", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"])
    pretrained = st.sidebar.checkbox("Use pretrained model")
    num_epochs = st.sidebar.number_input("Enter number of epochs:", min_value=1, value=20)
    img_size = st.sidebar.number_input("Enter image size:", min_value=32, value=416)
    batch_size = st.sidebar.number_input("Enter batch size:", min_value=1, value=16)

    uploaded_train_data = st.file_uploader("Upload train dataset (ZIP file)", type=["zip"])
    uploaded_val_data = st.file_uploader("Upload validation dataset (ZIP file)", type=["zip"])

    if st.sidebar.button("Train Model") and uploaded_train_data and uploaded_val_data:
        # Create project folder
        #project_dir = f"/Users/nishagarg7/Downloads/3i- infotech/Automl/AutoML_Services/{username}/{project_name}"
        project_dir = f"{username}/{project_name}"
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)

        # Extract uploaded ZIP files and get full paths
        extracted_train_dirs = extract_zip(uploaded_train_data, project_dir)
        extracted_val_dirs = extract_zip(uploaded_val_data, project_dir)
        
        # Train model with full paths
        st.write("Training in progress...")
        yolo_at_obj = YOLO_AUTO_TRAIN()
        model_save_path = yolo_at_obj.training(username, extracted_val_dirs[0], extracted_train_dirs[0], class_names.split(','), num_classes, model_name, pretrained, project_name, "experiment_", username, num_epochs, img_size, batch_size)
        st.success(f"Model trained successfully! Model saved at: {model_save_path}")

if __name__ == "__main__":
    main()


