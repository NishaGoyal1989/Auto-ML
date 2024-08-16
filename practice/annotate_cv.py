import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os
import zipfile
from Util.cv_utils import YOLO_AUTO_TRAIN  # Assuming your YOLO_AUTO_TRAIN class is defined in yolo_auto_train.py

# Function to extract images from a ZIP file
def extract_images_from_zip(zip_file, extract_dir, folder_name):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    extract_dir = os.path.join(extract_dir, folder_name)

    extracted_files = [os.path.join(extract_dir, file) for file in os.listdir(extract_dir) if file.endswith((".jpg", ".jpeg", ".png"))]
    # st.write(extract_dir)
    return extracted_files

# Function to extract ZIP file and return the extracted directory's full path
def extract_zip(zip_file, extract_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    extracted_dirs = [os.path.join(extract_dir, d) for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
    return extracted_dirs

# Define the Streamlit app
def main():
    st.title("YOLO Model Training and Image Annotation App")

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
    pretrained = st.sidebar.checkbox("Use pretrained model")
    num_epochs = st.sidebar.number_input("Enter number of epochs:", min_value=1, value=20)
    img_size = st.sidebar.number_input("Enter image size:", min_value=32, value=416)
    batch_size = st.sidebar.number_input("Enter batch size:", min_value=1, value=16)

    upload_type = st.sidebar.selectbox("Select upload type:", ["Dataset", "Images to Annotate"])

    if upload_type == "Dataset":
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

    elif upload_type == "Images to Annotate":
        uploaded_folder = st.file_uploader("Upload images to annotate (ZIP file)", type=["zip"])

        if uploaded_folder is not None:
            # Get the current working directory
            cwd = os.getcwd()

            # Extract uploaded ZIP folder
            extracted_folder = os.path.join(cwd, "extracted_images")
            if not os.path.exists(extracted_folder):
                os.makedirs(extracted_folder)

            with open(os.path.join(extracted_folder, uploaded_folder.name), "wb") as f:
                f.write(uploaded_folder.getbuffer())

            # Extract images from the ZIP folder
            #folder_name = os.path.splitext(uploaded_folder.name)[0]
            
            
            folder_name = os.path.splitext(uploaded_folder.name)[0]
            
            # folder_name= os.path.join("train",folder_name)
            # if not os.path.exists(folder_name):
            #     os.makedirs(folder_name)
            st.write(os.path.join(extracted_folder, uploaded_folder.name))
            image_files = extract_images_from_zip(os.path.join(extracted_folder, uploaded_folder.name), extracted_folder, folder_name)

            image_index = st.session_state.get("image_index", 0)
            image_path = image_files[image_index]
            image = Image.open(image_path)

            # Display the image
            col1, col2 = st.columns(2)
            # col1.image(image, caption=f"Image {image_index + 1}/{len(image_files)}", use_column_width=True)

            # Sidebar input for bounding box annotation
            st.sidebar.header("Bounding Box Annotation")
            st.sidebar.write("Draw bounding boxes on the image below.")
            box_label = st.sidebar.text_input("Label for Box", "")

            # Create a canvas for drawing bounding boxes
            canvas = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Orange color with 30% opacity
                stroke_width=2,
                stroke_color="rgba(255, 165, 0, 0.6)",  # Orange color with 60% opacity
                background_image=image,
                drawing_mode="rect",
                key="canvas",
                height=300,
                width=300,
                update_streamlit=True,
            )

            # Initialize list to store annotations
            annotations = []
            if canvas.json_data["objects"]:
                st.sidebar.write("Bounding Box Coordinates and Labels:")
                for obj in canvas.json_data["objects"]:
                    if obj["type"] == "rect":
                        # Check if all keys are present in the obj dictionary
                        if all(key in obj for key in ['left', 'top', 'width', 'height']):
                            # Normalize box coordinates
                            left = obj["left"] / image.width
                            top = obj["top"] / image.height
                            width = obj["width"] / image.width
                            height = obj["height"] / image.height
                            annotations.append([box_label, left, top, width, height])
                            #annotations.append([box_label, obj['left'], obj['top'], obj['width'], obj['height']])
                            st.sidebar.write(f"Label: {box_label}, Coordinates: {obj['left']}, {obj['top']}, {obj['width']}, {obj['height']}")
                        else:
                            st.warning("Bounding box coordinates are incomplete.")
                    else:
                        st.warning("Unknown object type found in canvas data.")
        annotation_done = st.sidebar.button("Annotation Done")

            # Annotation
            # Save annotated image and move to the next image
        if st.button("Next Image") and image_index < len(image_files) - 1:
            # Save the annotations to a text file with the name of the image file
            image_name = os.path.basename(image_path)
            if (image_index + 1) % 3 == 0:  # Every third image goes to the "valid" folder
                valid_folder = os.path.join(extracted_folder, "valid","images")
                labels_valid_folder = os.path.join(extracted_folder, "valid","labels")
                if not os.path.exists(valid_folder):
                    os.makedirs(valid_folder)
                if not os.path.exists(labels_valid_folder):
                    os.makedirs(labels_valid_folder)
                image.save(os.path.join(valid_folder, image_name))

                # Save the annotation text file in the "labels_valid" folder
                annotation_text_file = os.path.splitext(image_name)[0] + ".txt"
                annotation_text_path = os.path.join(labels_valid_folder, annotation_text_file)
                with open(annotation_text_path, "w") as annotation_file:
                    for annotation in annotations:
                        annotation_file.write(f"{annotation[0]} {annotation[1]} {annotation[2]} {annotation[3]} {annotation[4]}\n")
            else:
                train_folder = os.path.join(extracted_folder, "train","images")
                labels_train_folder = os.path.join(extracted_folder, "train","labels")
                if not os.path.exists(train_folder):
                    os.makedirs(train_folder)
                if not os.path.exists(labels_train_folder):
                    os.makedirs(labels_train_folder)
                image.save(os.path.join(train_folder, image_name))

                # Save the annotation text file in the "labels_train" folder
                annotation_text_file = os.path.splitext(image_name)[0] + ".txt"
                annotation_text_path = os.path.join(labels_train_folder, annotation_text_file)
                with open(annotation_text_path, "w") as annotation_file:
                    for annotation in annotations:
                        annotation_file.write(f"{annotation[0]} {annotation[1]} {annotation[2]} {annotation[3]} {annotation[4]}\n")

            # labels_folder = os.path.join(extracted_folder,"train", "labels")
            # if not os.path.exists(labels_folder):
            #     os.makedirs(labels_folder)
            
            # text_file_path = os.path.join(labels_folder, f"{os.path.splitext(image_name)[0]}.txt")
            # with open(text_file_path, "w") as text_file:
            #     for annotation in annotations:
            #         text_file.write(f"{annotation[0]} {annotation[1]} {annotation[2]} {annotation[3]} {annotation[4]}\n")
            # st.info(f"Bounding box annotations saved at: {text_file_path}")
            # # Determine if the image should be saved in the "train" or "valid" folder
            # if (image_index + 1) % 3 == 0:  # Every third image goes to the "valid" folder
            #     valid_folder = os.path.join(extracted_folder, "valid","images")
            #     if not os.path.exists(valid_folder):
            #         os.makedirs(valid_folder)
            #     image.save(os.path.join(valid_folder, image_name))
            # else:
            #     train_folder = os.path.join(extracted_folder, "train","images")
            #     if not os.path.exists(train_folder):
            #         os.makedirs(train_folder)
            #     image.save(os.path.join(train_folder, image_name))
            # # Move the annotated image to the "train" folder
            # train_folder = os.path.join(extracted_folder, "train","images")
            # if not os.path.exists(train_folder):
            #     os.makedirs(train_folder)
            # image.save(os.path.join(train_folder, os.path.basename(image_path)))

            # Move to the next image
            image_index += 1
            st.session_state["image_index"] = image_index
                        # Annotation button to indicate completion
            
        if annotation_done:
            st.success("Annotation Done! You can now proceed to train the model.")
            # train_model= st.sidebar.button("Train Model 2")
            # st.write(train_model)
        # Train Model button enabled when annotation is done
            
            st.write("training in progress")                    
            #project_dir = f"/Users/nishagarg7/Downloads/3i- infotech/Automl/AutoML_Services/{username}/{project_name}"
            project_dir = f"{username}/{project_name}"
            if not os.path.exists(project_dir):
                os.makedirs(project_dir)

            # Extract uploaded ZIP files and get full paths
            # extracted_train_dirs = extract_zip(uploaded_train_data, project_dir)
            # extracted_val_dirs = extract_zip(uploaded_val_data, project_dir)
            extracted_train_dirs = os.path.join(extracted_folder, "train")
            extracted_val_dirs = os.path.join(extracted_folder, "valid")
            yolo_at_obj = YOLO_AUTO_TRAIN()
            model_save_path = yolo_at_obj.training(username, extracted_val_dirs, extracted_train_dirs, class_names.split(','), num_classes, model_name, pretrained, project_name, "experiment_", username, num_epochs, img_size, batch_size)
            st.success(f"Model trained successfully! Model saved at: {model_save_path}")
            # Add your YOLO model training code here
                # Example:
                # yolo_at_obj = YOLO_AUTO_TRAIN()
                # model_save_path = yolo_at_obj.training(username, extracted_val_dirs[0], extracted_train_dirs[0], class_names.split(','), num_classes, model_name, pretrained, project_name, "experiment_", username, num_epochs, img_size, batch_size)
                # st.success(f"Model trained successfully! Model saved at: {model_save_path}")
            
                
if __name__ == "__main__":
    main()
